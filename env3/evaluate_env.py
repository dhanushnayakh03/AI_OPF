import torch
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as pplot
import matplotlib.pyplot as plt
import time
import os
import sys

# Import model architecture
sys.path.append(os.path.abspath(".."))
from train_model import OPFSurrogate

ENV_NAME = "ENV 3: N-1 Contingency (Transmission Line Outage Stress Test)"
ENV_DESC = "Grid operating under an N-1 contingency (Line 4-5 out of service), testing topological robustness."

def load_pretrained_ai():
    v_min, v_max = 0.94, 1.10
    model = OPFSurrogate(input_dim=22, output_dim=4, v_min=v_min, v_max=v_max)
    model.load_state_dict(torch.load("../best_model.pth"))
    model.eval()
    X_mean = np.load("../X_mean.npy")
    X_std = np.load("../X_std.npy")
    return model, X_mean, X_std

def configure_grid():
    net = nw.case14()
    
    # Simulate N-1 contingency: Take transmission line 4 (connecting Bus 4 to Bus 5) out of service
    outage_line_idx = 4
    net.line.loc[outage_line_idx, 'in_service'] = False
    
    net.bus['min_vm_pu'] = 0.94
    net.bus['max_vm_pu'] = 1.10
    net.gen['min_vm_pu'] = 0.94
    net.gen['max_vm_pu'] = 1.10
    net.gen['min_p_mw'] = net.gen['p_mw']
    net.gen['max_p_mw'] = net.gen['p_mw']
    net.gen['controllable'] = True
    net.ext_grid['min_q_mvar'] = -500.0
    net.ext_grid['max_q_mvar'] = 500.0
    net.ext_grid['min_p_mw'] = 0.0
    net.ext_grid['max_p_mw'] = 2000.0
    
    net.poly_cost.drop(net.poly_cost.index, inplace=True)
    net.pwl_cost.drop(net.pwl_cost.index, inplace=True)
    pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=1.0)
    return net, outage_line_idx

def run_environment_test(num_scenarios=200):
    print("="*65)
    print(f"   {ENV_NAME}")
    print(f"   {ENV_DESC}")
    print("="*65)
    
    model, X_mean, X_std = load_pretrained_ai()
    net, outage_line_idx = configure_grid()
    load_bus_order = net.load.sort_values('bus').index.tolist()
    
    base_p_mw = nw.case14().load.p_mw.copy()
    base_q_mvar = nw.case14().load.q_mvar.copy()
    
    losses_base = []
    losses_opf = []
    losses_ai = []
    times_opf = []
    times_ai = []
    constraint_violations = 0
    
    rep_bus_v_base = None
    rep_bus_v_opf = None
    rep_bus_v_ai = None
    rep_loading_base = None
    rep_loading_ai = None
    
    np.random.seed(303)
    valid_scenarios = 0
    
    for s in range(num_scenarios):
        scale_p = np.random.uniform(0.85, 1.15, size=len(net.load))
        scale_q = np.random.uniform(0.85, 1.15, size=len(net.load))
        
        net.load.loc[:, 'p_mw'] = base_p_mw * scale_p
        net.load.loc[:, 'q_mvar'] = base_q_mvar * scale_q
        
        # 1. Baseline Power Flow (Nominal 1.0 pu)
        net.gen.loc[:, 'vm_pu'] = [1.0, 1.0, 1.0, 1.0]
        try:
            pp.runpp(net, enforce_q_lims=True)
            p_loss_base = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
        except Exception:
            continue
            
        # 2. AC-OPF Benchmark under Contingency
        t0 = time.perf_counter()
        try:
            pp.runopp(net, verbose=False)
            t_opf = (time.perf_counter() - t0) * 1000.0
            if not net.OPF_converged:
                continue
            opt_vg = net.res_gen.vm_pu.values.copy()
            net.gen.loc[:, 'vm_pu'] = opt_vg
            pp.runpp(net, enforce_q_lims=True)
            p_loss_opf = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
        except Exception:
            continue
            
        # 3. Pre-Trained AI Surrogate
        u_raw = []
        for l_idx in load_bus_order:
            u_raw.append(net.load.loc[l_idx, 'p_mw'])
        for l_idx in load_bus_order:
            u_raw.append(net.load.loc[l_idx, 'q_mvar'])
        u_raw = np.array(u_raw)
        u_scaled = (u_raw - X_mean) / X_std
        
        t0 = time.perf_counter()
        with torch.no_grad():
            ai_pred_vg = model(torch.tensor(u_scaled, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        t_ai = (time.perf_counter() - t0) * 1000.0
        
        net.gen.loc[:, 'vm_pu'] = ai_pred_vg
        try:
            pp.runpp(net, enforce_q_lims=True)
            p_loss_ai = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
            v_viol = np.any(net.res_bus.vm_pu < 0.939) or np.any(net.res_bus.vm_pu > 1.101)
            line_viol = np.any(net.res_line.loading_percent > 100.0)
            if v_viol or line_viol:
                constraint_violations += 1
        except Exception:
            p_loss_ai = np.nan
            constraint_violations += 1
            
        losses_base.append(p_loss_base)
        losses_opf.append(p_loss_opf)
        losses_ai.append(p_loss_ai)
        times_opf.append(t_opf)
        times_ai.append(t_ai)
        
        if valid_scenarios == 0:
            rep_bus_v_base = net.res_bus.vm_pu.values.copy()
            rep_loading_base = net.res_line.loading_percent.values.copy()
            net.gen.loc[:, 'vm_pu'] = opt_vg
            pp.runpp(net, enforce_q_lims=True)
            rep_bus_v_opf = net.res_bus.vm_pu.values.copy()
            net.gen.loc[:, 'vm_pu'] = ai_pred_vg
            pp.runpp(net, enforce_q_lims=True)
            rep_bus_v_ai = net.res_bus.vm_pu.values.copy()
            rep_loading_ai = net.res_line.loading_percent.values.copy()
            
        valid_scenarios += 1
        
    losses_base = np.array(losses_base)
    losses_opf = np.array(losses_opf)
    losses_ai = np.array(losses_ai)
    
    avg_base = np.mean(losses_base)
    avg_opf = np.mean(losses_opf)
    avg_ai = np.mean(losses_ai)
    
    loss_red = ((losses_base - losses_ai) / losses_base) * 100.0
    opt_gap = ((losses_ai - losses_opf) / losses_opf) * 100.0
    avg_t_opf = np.mean(times_opf)
    avg_t_ai = np.mean(times_ai)
    
    print("\n" + "-"*65)
    print(f"Results Summary across {valid_scenarios} N-1 Contingency Scenarios:")
    print(f"  - Average Baseline Loss (1.0 pu): {avg_base:.4f} MW")
    print(f"  - Average AC-OPF Optimal Loss:    {avg_opf:.4f} MW")
    print(f"  - Average AI Surrogate Loss:      {avg_ai:.4f} MW")
    print(f"  - Loss Reduction:                 {np.mean(loss_red):.2f}% (Saved {avg_base - avg_ai:.2f} MW)")
    print(f"  - AI Optimality Gap vs OPF:       {np.mean(opt_gap):.3f}%")
    print(f"  - Constraint Violation Rate:      {(constraint_violations/valid_scenarios)*100:.2f}%")
    print(f"  - Average Solver Execution Time:  {avg_t_opf:.2f} ms")
    print(f"  - Average AI Inference Time:      {avg_t_ai:.4f} ms ({avg_t_opf/avg_t_ai:.1f}x speedup)")
    print("-"*65 + "\n")
    
    # Generate Visual Figures
    os.makedirs("figures", exist_ok=True)
    
    # 1. Topology with Outage Line Highlighted
    # ================================================================
    # 1. CLEAN IEEE-14 TOPOLOGY (N-1 CONTINGENCY)
    # ================================================================
    fig, ax = plt.subplots(figsize=(14, 9))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ------------------------------------------------
    # Carefully chosen coordinates
    # ------------------------------------------------
    # (pandapower uses 0-based indices internally)
    pos = {
        0:  (0.0,  2.5),    # Bus 1
        1:  (2.2,  3.8),    # Bus 2
        2:  (2.2,  1.2),    # Bus 3
        3:  (4.5,  2.5),    # Bus 4
        4:  (6.8,  3.8),    # Bus 5
        5:  (6.8,  1.2),    # Bus 6
        6:  (9.2,  3.8),    # Bus 7
        7:  (9.2,  1.2),    # Bus 8
        8:  (11.5, 3.8),    # Bus 9
        9:  (11.5, 1.2),    # Bus 10
        10: (13.8, 3.8),    # Bus 11
        11: (13.8, 1.2),    # Bus 12
        12: (16.0, 3.8),    # Bus 13
        13: (16.0, 1.2),    # Bus 14
    }

    # ------------------------------------------------
    # Helper: draw a clean branch
    # ------------------------------------------------
    def draw_branch(bus1, bus2, color="#64748b",
                    lw=2.2, linestyle="-",
                    curve=0.0, zorder=1):

        x1, y1 = pos[bus1]
        x2, y2 = pos[bus2]

        if curve == 0:
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=lw,
                linestyle=linestyle,
                solid_capstyle="round",
                zorder=zorder
            )
        else:
            from matplotlib.path import Path
            from matplotlib.patches import PathPatch

            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)

            # perpendicular displacement
            px = -dy / length * curve
            py = dx / length * curve

            cx = (x1 + x2) / 2 + px
            cy = (y1 + y2) / 2 + py

            path = Path(
                [
                    (x1, y1),
                    (cx, cy),
                    (x2, y2)
                ],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            )

            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor=color,
                linewidth=lw,
                linestyle=linestyle,
                zorder=zorder
            )

            ax.add_patch(patch)


    # ------------------------------------------------
    # Transmission lines
    # ------------------------------------------------
    # Explicitly draw the IEEE-14 branches.
    # Small curves are used only where they improve readability.

    branches = [
        (0, 1,  0.0),    # 1-2
        (0, 4,  0.45),   # 1-5

        (1, 2,  0.0),    # 2-3
        (1, 3,  0.0),    # 2-4
        (1, 4, -0.35),   # 2-5

        (2, 3,  0.0),    # 3-4
        (3, 4,  0.0),    # 4-5

        (4, 5,  0.0),    # 5-6

        (5, 10,  0.45),  # 6-11
        (5, 11,  0.0),   # 6-12
        (5, 12, -0.45),  # 6-13

        (6, 7,  0.0),    # 7-8
        (6, 8,  0.0),    # 7-9

        (8, 9,  0.0),    # 9-10
        (8, 13, 0.45),   # 9-14

        (9, 10, 0.0),    # 10-11

        (11, 12, 0.0),   # 12-13
        (12, 13, 0.0),   # 13-14
    ]

    # In env3: Identify out-of-service branches (N-1 contingency on Line 4-5)
    outage_lines = set()
    for _, line in net.line[net.line.in_service == False].iterrows():
        fb, tb = int(line.from_bus), int(line.to_bus)
        outage_lines.add((min(fb, tb), max(fb, tb)))

    for b1, b2, curve in branches:
        pair = (min(b1, b2), max(b1, b2))
        if pair in outage_lines:
            draw_branch(b1, b2, color="#ef4444", lw=2.5, linestyle="--", curve=curve, zorder=4)
        else:
            draw_branch(b1, b2, curve=curve)


    # ------------------------------------------------
    # Transformers
    # ------------------------------------------------
    # Draw transformers over the corresponding branches.
    # Green is deliberately thinner than before.

    transformers = [
        (3, 6),   # 4-7
        (3, 8),   # 4-9
        (4, 5),   # 5-6
    ]

    for b1, b2 in transformers:

        x1, y1 = pos[b1]
        x2, y2 = pos[b2]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#10b981",
            linewidth=3.5,
            zorder=3
        )

        # Transformer marker
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2

        ax.scatter(
            xm, ym,
            s=110,
            facecolor="white",
            edgecolor="#10b981",
            linewidth=2,
            zorder=5
        )

        ax.text(
            xm, ym,
            "T",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#059669",
            zorder=6
        )


    # ------------------------------------------------
    # Determine buses containing generators and loads
    # ------------------------------------------------
    ext_grid_buses = set(net.ext_grid.bus.astype(int))
    gen_buses = set(net.gen.bus.astype(int))
    load_buses = set(net.load.bus.astype(int))


    # ------------------------------------------------
    # Bus nodes
    # ------------------------------------------------
    for bus, (x, y) in pos.items():

        ax.scatter(
            x, y,
            s=700,
            facecolor="white",
            edgecolor="#2563eb",
            linewidth=3,
            zorder=10
        )

        ax.text(
            x, y,
            str(bus + 1),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#111827",
            zorder=11
        )


    # ------------------------------------------------
    # Generator symbols
    # ------------------------------------------------
    for bus in gen_buses:

        x, y = pos[bus]

        # connector
        ax.plot(
            [x, x],
            [y + 0.40, y + 0.75],
            color="#f59e0b",
            linewidth=1.5,
            zorder=7
        )

        ax.scatter(
            x,
            y + 0.95,
            s=300,
            marker="^",
            facecolor="#f59e0b",
            edgecolor="#b45309",
            linewidth=1.5,
            zorder=12
        )

        ax.text(
            x,
            y + 0.95,
            "G",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            zorder=13
        )


    # ------------------------------------------------
    # Load symbols
    # ------------------------------------------------
    for bus in load_buses:

        x, y = pos[bus]

        ax.plot(
            [x, x],
            [y - 0.40, y - 0.75],
            color="#8b5cf6",
            linewidth=1.5,
            zorder=7
        )

        ax.scatter(
            x,
            y - 0.95,
            s=280,
            marker="v",
            facecolor="#8b5cf6",
            edgecolor="#6d28d9",
            linewidth=1.5,
            zorder=12
        )

        ax.text(
            x,
            y - 0.95,
            "L",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            zorder=13
        )


    # ------------------------------------------------
    # External grid
    # ------------------------------------------------
    for bus in ext_grid_buses:

        x, y = pos[bus]

        ax.plot(
            [x - 0.9, x - 0.4],
            [y, y],
            color="#dc2626",
            linewidth=3,
            zorder=7
        )

        ax.scatter(
            x - 1.05,
            y,
            s=280,
            marker="s",
            facecolor="#dc2626",
            edgecolor="#991b1b",
            linewidth=1.5,
            zorder=12
        )

        ax.text(
            x - 1.05,
            y,
            "G",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            zorder=13
        )


    # ------------------------------------------------
    # Title
    # ------------------------------------------------
    ax.set_title(
        "IEEE 14-Bus Power System Topology",
        fontsize=20,
        fontweight="bold",
        color="#111827",
        pad=28
    )

    ax.text(
        0.5,
        1.015,
        "Transmission network structure (Line 4-5 Outage)",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="#64748b"
    )


    # ------------------------------------------------
    # Legend
    # ------------------------------------------------
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#2563eb",
            markeredgewidth=2.5,
            markersize=12,
            label="Bus"
        ),

        Line2D(
            [0], [0],
            marker="^",
            color="none",
            markerfacecolor="#f59e0b",
            markeredgecolor="#b45309",
            markersize=10,
            label="Generator"
        ),

        Line2D(
            [0], [0],
            marker="v",
            color="none",
            markerfacecolor="#8b5cf6",
            markeredgecolor="#6d28d9",
            markersize=10,
            label="Load"
        ),

        Line2D(
            [0], [0],
            color="#64748b",
            linewidth=2.2,
            label="Transmission Line"
        ),

        Line2D(
            [0], [0],
            color="#ef4444",
            linewidth=2.5,
            linestyle="--",
            label="Outage Line (Tripped)"
        ),

        Line2D(
            [0], [0],
            color="#10b981",
            linewidth=3.5,
            label="Transformer"
        ),

        Line2D(
            [0], [0],
            color="#dc2626",
            linewidth=3,
            label="External Grid"
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        frameon=False,
        fontsize=10
    )


    # ------------------------------------------------
    # Final formatting
    # ------------------------------------------------
    ax.set_xlim(-1.8, 17.0)
    ax.set_ylim(-1.7, 5.2)

    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

    fig.savefig(
        "figures/fig1_topology.png",
        dpi=350,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)
    
    # 2. Loss Comparison
    plt.figure(figsize=(10, 5))
    s_idx = np.arange(min(40, len(losses_base)))
    plt.plot(s_idx, losses_base[:len(s_idx)], 'o--', color='#ef4444', label='Nominal Baseline (1.0 pu)', alpha=0.7)
    plt.plot(s_idx, losses_opf[:len(s_idx)], 's-', color='#10b981', label='AC-OPF Benchmark (Contingency)', alpha=0.9)
    plt.plot(s_idx, losses_ai[:len(s_idx)], '^:', color='#3b82f6', label='Trained AI Surrogate', alpha=0.9, linewidth=2)
    plt.xlabel('Contingency Scenario Index', fontsize=11)
    plt.ylabel('Transmission Loss (MW)', fontsize=11)
    plt.title('Figure 2: Transmission Loss under N-1 Contingency (Baseline vs OPF vs AI)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/fig2_loss_comparison.png", dpi=300)
    plt.close()
    
    # 3. Scatter AI vs OPF
    plt.figure(figsize=(7, 7))
    plt.scatter(losses_opf, losses_ai, color='#3b82f6', alpha=0.7, edgecolors='k', s=45, label='N-1 Scenarios')
    min_v = min(np.min(losses_opf), np.min(losses_ai)) * 0.98
    max_v = max(np.max(losses_opf), np.max(losses_ai)) * 1.02
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Ideal Equivalence (y = x)')
    plt.xlim(min_v, max_v)
    plt.ylim(min_v, max_v)
    plt.xlabel('AC-OPF Loss (MW)', fontsize=11)
    plt.ylabel('AI Surrogate Loss (MW)', fontsize=11)
    plt.title('Figure 3: AI Robustness under N-1 Contingency vs AC-OPF', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/fig3_ai_vs_opf_scatter.png", dpi=300)
    plt.close()
    
    # 4. Voltage Profile
    plt.figure(figsize=(9, 5))
    buses = np.arange(1, len(net.bus) + 1)
    plt.plot(buses, rep_bus_v_base, 'o--', color='#ef4444', label='Nominal Baseline', linewidth=1.5)
    plt.plot(buses, rep_bus_v_opf, 's-', color='#10b981', label='AC-OPF Optimal', linewidth=2)
    plt.plot(buses, rep_bus_v_ai, '^:', color='#3b82f6', label='Trained AI Surrogate', linewidth=2)
    plt.axhline(0.94, color='red', linestyle='--', alpha=0.7, label='V_min (0.94 pu)')
    plt.axhline(1.10, color='red', linestyle='--', alpha=0.7, label='V_max (1.10 pu)')
    plt.xticks(buses)
    plt.xlabel('Bus Number', fontsize=11)
    plt.ylabel('Voltage (p.u.)', fontsize=11)
    plt.title('Figure 4: Bus Voltage Profile during N-1 Contingency Condition', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/fig4_voltage_profile.png", dpi=300)
    plt.close()
    
    # 5. Line Loading
    plt.figure(figsize=(9, 5))
    lines = np.arange(1, len(net.line) + 1)
    bar_w = 0.38
    plt.bar(lines - bar_w/2, rep_loading_base, width=bar_w, label='Baseline Loading', color='#f87171', alpha=0.85)
    plt.bar(lines + bar_w/2, rep_loading_ai, width=bar_w, label='AI Controlled Loading', color='#3b82f6', alpha=0.85)
    plt.axhline(100.0, color='red', linestyle='--', label='100% Thermal Rating')
    plt.xticks(lines)
    plt.xlabel('Transmission Line', fontsize=11)
    plt.ylabel('Loading (%)', fontsize=11)
    plt.title('Figure 5: Branch Loading during N-1 Line Outage', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/fig5_line_loading.png", dpi=300)
    plt.close()
    
    # 6. Computation Time
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['AC-OPF Solver', 'AI Inference'], [avg_t_opf, avg_t_ai], color=['#f59e0b', '#3b82f6'], width=0.5, edgecolor='k', alpha=0.85)
    plt.yscale('log')
    plt.ylabel('Execution Time (ms) - Log Scale', fontsize=11)
    plt.title('Figure 6: Execution Speed Comparison', fontsize=12, fontweight='bold')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h * 1.3, f'{h:.3f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6, which='both')
    plt.tight_layout()
    plt.savefig("figures/fig6_computation_time.png", dpi=300)
    plt.close()
    print("[ENV3] All 6 figures saved to 'figures/'.")

if __name__ == "__main__":
    run_environment_test()
