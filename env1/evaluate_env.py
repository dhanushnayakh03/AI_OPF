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

ENV_NAME = "ENV 1: Peak Demand & Heavy Load Stress Test"
ENV_DESC = "Grid operating under heavy peak loading (120% to 150% of nominal load)."

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
    return net

def run_environment_test(num_scenarios=200):
    print("="*65)
    print(f"   {ENV_NAME}")
    print(f"   {ENV_DESC}")
    print("="*65)
    
    model, X_mean, X_std = load_pretrained_ai()
    net = configure_grid()
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
    
    np.random.seed(101)
    valid_scenarios = 0
    
    for s in range(num_scenarios):
        # Heavy peak load scaling: 1.2 to 1.5x
        scale_p = np.random.uniform(1.2, 1.5, size=len(net.load))
        scale_q = np.random.uniform(1.2, 1.5, size=len(net.load))
        
        net.load.loc[:, 'p_mw'] = base_p_mw * scale_p
        net.load.loc[:, 'q_mvar'] = base_q_mvar * scale_q
        
        # 1. Baseline Power Flow (Nominal 1.0 pu)
        net.gen.loc[:, 'vm_pu'] = [1.0, 1.0, 1.0, 1.0]
        try:
            pp.runpp(net, enforce_q_lims=True)
            p_loss_base = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
        except Exception:
            continue
            
        # 2. AC-OPF Benchmark
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
    print(f"Results Summary across {valid_scenarios} Heavy Load Scenarios:")
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
    
    # 1. Topology
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    pplot.create_generic_coordinates(net, overwrite=True)
    import json
    if not hasattr(net, 'bus_geodata'):
        net.bus_geodata = pd.DataFrame(index=net.bus.index, columns=['x', 'y'])
        if 'geo' in net.bus.columns:
            for idx in net.bus.index:
                if not pd.isna(net.bus.loc[idx, 'geo']):
                    geo_dict = json.loads(net.bus.loc[idx, 'geo'])
                    net.bus_geodata.loc[idx, 'x'] = geo_dict['coordinates'][0]
                    net.bus_geodata.loc[idx, 'y'] = geo_dict['coordinates'][1]
    bc = pplot.create_bus_collection(net, size=0.2, color='#3b82f6', zorder=10)
    lc = pplot.create_line_collection(net, color='#9ca3af', linewidths=2., use_bus_geodata=True)
    tc = pplot.create_trafo_collection(net, size=0.3, color='#10b981')
    ext = pplot.create_ext_grid_collection(net, size=0.4, orientation=-1.5, color='#ef4444')
    gen = pplot.create_gen_collection(net, size=0.3, color='#f59e0b')
    load = pplot.create_load_collection(net, size=0.3, color='#8b5cf6')
    pplot.draw_collections([lc, tc, bc, ext, gen, load], ax=ax)
    for i, row in net.bus_geodata.iterrows():
        ax.annotate(str(i), (row.x, row.y + 0.3), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1f2937')
    ax.set_title("Figure 1: IEEE-14 Topology under Heavy Peak Load (120-150%)", fontsize=14, fontweight='bold', color='#111827', pad=15)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig("figures/fig1_topology.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Loss Comparison
    plt.figure(figsize=(10, 5))
    s_idx = np.arange(min(40, len(losses_base)))
    plt.plot(s_idx, losses_base[:len(s_idx)], 'o--', color='#ef4444', label='Nominal Baseline (1.0 pu)', alpha=0.7)
    plt.plot(s_idx, losses_opf[:len(s_idx)], 's-', color='#10b981', label='AC-OPF Benchmark', alpha=0.9)
    plt.plot(s_idx, losses_ai[:len(s_idx)], '^:', color='#3b82f6', label='Trained AI Surrogate', alpha=0.9, linewidth=2)
    plt.xlabel('Heavy Load Scenario Index', fontsize=11)
    plt.ylabel('Transmission Loss (MW)', fontsize=11)
    plt.title('Figure 2: Transmission Loss under Heavy Loading (Baseline vs OPF vs AI)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/fig2_loss_comparison.png", dpi=300)
    plt.close()
    
    # 3. Scatter AI vs OPF
    plt.figure(figsize=(7, 7))
    plt.scatter(losses_opf, losses_ai, color='#3b82f6', alpha=0.7, edgecolors='k', s=45, label='Peak Scenarios')
    min_v = min(np.min(losses_opf), np.min(losses_ai)) * 0.98
    max_v = max(np.max(losses_opf), np.max(losses_ai)) * 1.02
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Ideal Equivalence (y = x)')
    plt.xlim(min_v, max_v)
    plt.ylim(min_v, max_v)
    plt.xlabel('AC-OPF Loss (MW)', fontsize=11)
    plt.ylabel('AI Surrogate Loss (MW)', fontsize=11)
    plt.title('Figure 3: AI Generalization vs AC-OPF (Heavy Load)', fontsize=12, fontweight='bold')
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
    plt.title('Figure 4: Bus Voltage Profile under Heavy Peak Demand', fontsize=12, fontweight='bold')
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
    plt.title('Figure 5: Branch Loading Reduction with AI under Peak Demand', fontsize=12, fontweight='bold')
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
    print("[ENV1] All 6 figures saved to 'figures/'.")

if __name__ == "__main__":
    run_environment_test()
