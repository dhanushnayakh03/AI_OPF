import torch
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as pplot
import matplotlib.pyplot as plt
import time
import os

from train_model import OPFSurrogate
from data_generation import create_and_configure_network

def load_ai_model(model_path="best_model.pth", v_min=0.94, v_max=1.10):
    model = OPFSurrogate(input_dim=22, output_dim=4, v_min=v_min, v_max=v_max)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    X_mean = np.load("X_mean.npy")
    X_std = np.load("X_std.npy")
    return model, X_mean, X_std

def evaluate_all(v_min=0.94, v_max=1.10):
    print("============================================================")
    print("     EVALUATION: BASELINE vs AC-OPF vs AI SURROGATE")
    print("============================================================")
    
    # 1. Load trained AI and normalizer
    model, X_mean, X_std = load_ai_model(v_min=v_min, v_max=v_max)
    
    # 2. Load test data
    X_test_raw = np.load("X_test_raw.npy")
    y_test = np.load("y_test.npy")
    X_test_scaled = (X_test_raw - X_mean) / X_std
    
    num_test = len(X_test_raw)
    print(f"Loaded {num_test} unseen test scenarios.\n")
    
    net = create_and_configure_network(v_min=v_min, v_max=v_max)
    load_bus_order = net.load.sort_values('bus').index.tolist()
    
    # Storage for results
    losses_flat_base = []
    losses_case14_base = []
    losses_opf = []
    losses_ai = []
    
    times_opf = []
    times_ai = []
    
    constraint_violations = 0
    
    # Profile history for representative plotting
    rep_bus_v_base = None
    rep_bus_v_opf = None
    rep_bus_v_ai = None
    
    rep_line_loading_base = None
    rep_line_loading_ai = None
    
    for i in range(num_test):
        # Set loads for this scenario
        p_loads = X_test_raw[i, :11]
        q_loads = X_test_raw[i, 11:22]
        
        for idx, l_idx in enumerate(load_bus_order):
            net.load.loc[l_idx, 'p_mw'] = p_loads[idx]
            net.load.loc[l_idx, 'q_mvar'] = q_loads[idx]
            
        # ----------------------------------------------------
        # Strategy A1: NOMINAL FLAT BASELINE (1.0 pu)
        # ----------------------------------------------------
        net.gen.loc[:, 'vm_pu'] = [1.0, 1.0, 1.0, 1.0]
        pp.runpp(net, enforce_q_lims=True)
        p_loss_flat = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
        losses_flat_base.append(p_loss_flat)
        
        if i == 0:
            rep_bus_v_base = net.res_bus.vm_pu.values.copy()
            rep_line_loading_base = net.res_line.loading_percent.values.copy()

        # ----------------------------------------------------
        # Strategy A2: Standard Case14 Setpoints Baseline
        # ----------------------------------------------------
        net.gen.loc[:, 'vm_pu'] = nw.case14().gen.vm_pu.values
        pp.runpp(net, enforce_q_lims=True)
        p_loss_c14 = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
        losses_case14_base.append(p_loss_c14)
            
        # ----------------------------------------------------
        # Strategy B: AC-OPF Benchmark
        # ----------------------------------------------------
        t0_opf = time.perf_counter()
        pp.runopp(net, verbose=False)
        t1_opf = time.perf_counter()
        times_opf.append((t1_opf - t0_opf) * 1000.0) # in ms
        
        opt_vg = net.res_gen.vm_pu.values.copy()
        
        # Verify OPF physical loss via AC power flow
        net.gen.loc[:, 'vm_pu'] = opt_vg
        pp.runpp(net, enforce_q_lims=True)
        p_loss_opf = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
        losses_opf.append(p_loss_opf)
        
        if i == 0:
            rep_bus_v_opf = net.res_bus.vm_pu.values.copy()
            
        # ----------------------------------------------------
        # Strategy C: AI Surrogate Prediction
        # ----------------------------------------------------
        t0_ai = time.perf_counter()
        x_tensor = torch.tensor(X_test_scaled[i:i+1], dtype=torch.float32)
        with torch.no_grad():
            ai_pred_vg = model(x_tensor).numpy()[0]
        t1_ai = time.perf_counter()
        times_ai.append((t1_ai - t0_ai) * 1000.0) # in ms
        
        # Insert AI predictions into physical network & run AC power flow
        net.gen.loc[:, 'vm_pu'] = ai_pred_vg
        try:
            pp.runpp(net, enforce_q_lims=True)
            p_loss_ai = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
            
            # Check constraints
            v_viol = np.any(net.res_bus.vm_pu < (v_min - 1e-3)) or np.any(net.res_bus.vm_pu > (v_max + 1e-3))
            line_viol = np.any(net.res_line.loading_percent > 100.0)
            if v_viol or line_viol:
                constraint_violations += 1
        except Exception:
            p_loss_ai = np.nan
            constraint_violations += 1
            
        losses_ai.append(p_loss_ai)
        
        if i == 0:
            rep_bus_v_ai = net.res_bus.vm_pu.values.copy()
            rep_line_loading_ai = net.res_line.loading_percent.values.copy()
            
    losses_flat_base = np.array(losses_flat_base)
    losses_case14_base = np.array(losses_case14_base)
    losses_opf = np.array(losses_opf)
    losses_ai = np.array(losses_ai)
    
    # ----------------------------------------------------
    # Calculate Quantitative Performance Metrics
    # ----------------------------------------------------
    avg_loss_flat = np.mean(losses_flat_base)
    avg_loss_c14 = np.mean(losses_case14_base)
    avg_loss_opf = np.mean(losses_opf)
    avg_loss_ai = np.mean(losses_ai)
    
    loss_reduction_flat_pct = ((losses_flat_base - losses_ai) / losses_flat_base) * 100.0
    loss_reduction_c14_pct = ((losses_case14_base - losses_ai) / losses_case14_base) * 100.0
    optimality_gap_pct = ((losses_ai - losses_opf) / losses_opf) * 100.0
    
    avg_loss_reduction_flat = np.mean(loss_reduction_flat_pct)
    avg_loss_reduction_c14 = np.mean(loss_reduction_c14_pct)
    avg_optimality_gap = np.mean(optimality_gap_pct)
    violation_rate = (constraint_violations / num_test) * 100.0
    
    avg_t_opf = np.mean(times_opf)
    avg_t_ai = np.mean(times_ai)
    speedup = avg_t_opf / avg_t_ai if avg_t_ai > 0 else float('inf')
    
    print("============================================================")
    print("                OVERALL TEST SET SUMMARY")
    print("============================================================")
    print(f"Average Flat Baseline Loss (1.0 pu): {avg_loss_flat:.4f} MW")
    print(f"Average Case14 Baseline Loss:        {avg_loss_c14:.4f} MW")
    print(f"Average AC-OPF Benchmark Loss:       {avg_loss_opf:.4f} MW")
    print(f"Average AI Surrogate Loss:           {avg_loss_ai:.4f} MW")
    print("------------------------------------------------------------")
    print(f"Loss Reduction vs Flat Baseline:     {avg_loss_reduction_flat:.2f}%  (Saves {avg_loss_flat - avg_loss_ai:.2f} MW)")
    print(f"Loss Reduction vs Case14 Baseline:   {avg_loss_reduction_c14:.2f}%")
    print(f"Average AI Optimality Gap:           {avg_optimality_gap:.3f}%")
    print(f"Constraint Violation Rate:           {violation_rate:.2f}%")
    print("------------------------------------------------------------")
    print(f"Average AC-OPF Time:                 {avg_t_opf:.2f} ms")
    print(f"Average AI Inference Time:           {avg_t_ai:.4f} ms")
    print(f"Speedup Factor:                      {speedup:.1f}x faster")
    print("============================================================\n")
    
    # ----------------------------------------------------
    # Generate 6 Required Figures
    # ----------------------------------------------------
    os.makedirs("figures", exist_ok=True)
    
    # FIGURE 1: Network Topology
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    try:
        if net.bus_geodata.empty:
            pplot.create_generic_coordinates(net, overwrite=True)
    except AttributeError:
        pplot.create_generic_coordinates(net, overwrite=True)
    bc = pplot.create_bus_collection(net, size=0.2, color='#3b82f6', zorder=10)
    lc = pplot.create_line_collection(net, color='#9ca3af', linewidths=2., use_bus_geodata=True)
    tc = pplot.create_trafo_collection(net, size=0.3, color='#10b981')
    ext = pplot.create_ext_grid_collection(net, size=0.4, orientation=-1.5, color='#ef4444')
    gen = pplot.create_gen_collection(net, size=0.3, color='#f59e0b')
    load = pplot.create_load_collection(net, size=0.3, color='#8b5cf6')
    pplot.draw_collections([lc, tc, bc, ext, gen, load], ax=ax)
    for i, row in net.bus_geodata.iterrows():
        ax.annotate(str(i), (row.x, row.y + 0.3), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1f2937')
    ax.set_title("Figure 1: IEEE-14 Bus Standard Transmission System Topology", fontsize=15, fontweight='bold', color='#111827', pad=15)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig("figures/fig1_topology.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved 'figures/fig1_topology.png'")
    
    # FIGURE 2: Baseline vs AI vs OPF Transmission Loss
    plt.figure(figsize=(10, 5))
    sample_indices = np.arange(min(40, num_test))
    plt.plot(sample_indices, losses_flat_base[:len(sample_indices)], 'o--', color='#ef4444', label='Nominal Flat Baseline (1.0 pu)', alpha=0.7)
    plt.plot(sample_indices, losses_opf[:len(sample_indices)], 's-', color='#10b981', label='AC-OPF Benchmark', alpha=0.9)
    plt.plot(sample_indices, losses_ai[:len(sample_indices)], '^:', color='#3b82f6', label='AI Surrogate', alpha=0.9, linewidth=2)
    plt.xlabel('Scenario Index', fontsize=12)
    plt.ylabel('Transmission Loss (MW)', fontsize=12)
    plt.title('Figure 2: Transmission Loss Comparison (Baseline vs AC-OPF vs AI)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/fig2_loss_comparison.png", dpi=300)
    plt.close()
    print("Saved 'figures/fig2_loss_comparison.png'")
    
    # FIGURE 3: P_loss_AI vs P_loss_OPF with y=x line
    plt.figure(figsize=(7, 7))
    plt.scatter(losses_opf, losses_ai, color='#3b82f6', alpha=0.7, edgecolors='k', s=45, label='Test Scenarios')
    min_v = min(np.min(losses_opf), np.min(losses_ai)) * 0.98
    max_v = max(np.max(losses_opf), np.max(losses_ai)) * 1.02
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Ideal Equivalence (y = x)')
    plt.xlim(min_v, max_v)
    plt.ylim(min_v, max_v)
    plt.xlabel('AC-OPF Transmission Loss (MW)', fontsize=12)
    plt.ylabel('AI Surrogate Transmission Loss (MW)', fontsize=12)
    plt.title('Figure 3: AI Predicted Loss vs AC-OPF Optimal Loss', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/fig3_ai_vs_opf_scatter.png", dpi=300)
    plt.close()
    print("Saved 'figures/fig3_ai_vs_opf_scatter.png'")
    
    # FIGURE 4: Bus Voltage Profile
    plt.figure(figsize=(9, 5))
    buses = np.arange(1, len(net.bus) + 1)
    plt.plot(buses, rep_bus_v_base, 'o--', color='#ef4444', label='Nominal Baseline (1.0 pu)', linewidth=1.5)
    plt.plot(buses, rep_bus_v_opf, 's-', color='#10b981', label='AC-OPF Optimal', linewidth=2)
    plt.plot(buses, rep_bus_v_ai, '^:', color='#3b82f6', label='AI Surrogate', linewidth=2)
    plt.axhline(v_min, color='red', linestyle='--', alpha=0.7, label=f'V_min ({v_min} pu)')
    plt.axhline(v_max, color='red', linestyle='--', alpha=0.7, label=f'V_max ({v_max} pu)')
    plt.xticks(buses)
    plt.xlabel('Bus Number', fontsize=12)
    plt.ylabel('Voltage Magnitude (p.u.)', fontsize=12)
    plt.title('Figure 4: Bus Voltage Profile for Unseen Operating Condition', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10, loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/fig4_voltage_profile.png", dpi=300)
    plt.close()
    print("Saved 'figures/fig4_voltage_profile.png'")
    
    # FIGURE 5: Line Loading Profile
    plt.figure(figsize=(9, 5))
    lines = np.arange(1, len(net.line) + 1)
    bar_width = 0.38
    plt.bar(lines - bar_width/2, rep_line_loading_base, width=bar_width, label='Baseline Loading', color='#f87171', alpha=0.85)
    plt.bar(lines + bar_width/2, rep_line_loading_ai, width=bar_width, label='AI Loading', color='#3b82f6', alpha=0.85)
    plt.axhline(100.0, color='red', linestyle='--', label='100% Thermal Rating Limit')
    plt.xticks(lines)
    plt.xlabel('Line Number', fontsize=12)
    plt.ylabel('Loading (%)', fontsize=12)
    plt.title('Figure 5: Branch Loading Comparison (Baseline vs AI Surrogate)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/fig5_line_loading.png", dpi=300)
    plt.close()
    print("Saved 'figures/fig5_line_loading.png'")
    
    # FIGURE 6: Computation Time Comparison
    plt.figure(figsize=(6, 5))
    methods = ['AC-OPF (Numerical)', 'AI Surrogate (Inference)']
    times = [avg_t_opf, avg_t_ai]
    colors = ['#f59e0b', '#3b82f6']
    bars = plt.bar(methods, times, color=colors, width=0.5, edgecolor='k', alpha=0.85)
    plt.yscale('log')
    plt.ylabel('Execution Time (ms) - Log Scale', fontsize=12)
    plt.title('Figure 6: Execution Speed Comparison', fontsize=13, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.3,
                 f'{height:.3f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6, which='both')
    plt.tight_layout()
    plt.savefig("figures/fig6_computation_time.png", dpi=300)
    plt.close()
    print("Saved 'figures/fig6_computation_time.png'")
    
    # ----------------------------------------------------
    # SECTION 17: Demonstration on a Completely Unseen Scenario
    # ----------------------------------------------------
    print("\n" + "="*60)
    print("   FINAL DEMONSTRATION: SINGLE UNSEEN OPERATING CONDITION")
    print("="*60)
    
    np.random.seed(999)
    fresh_net = create_and_configure_network(v_min=v_min, v_max=v_max)
    scale_p = np.random.uniform(0.8, 1.2, size=len(fresh_net.load))
    scale_q = np.random.uniform(0.8, 1.2, size=len(fresh_net.load))
    fresh_net.load.loc[:, 'p_mw'] = nw.case14().load.p_mw * scale_p
    fresh_net.load.loc[:, 'q_mvar'] = nw.case14().load.q_mvar * scale_q
    
    # 1. Baseline Power Flow (Nominal 1.0 pu)
    fresh_net.gen.loc[:, 'vm_pu'] = [1.0, 1.0, 1.0, 1.0]
    pp.runpp(fresh_net, enforce_q_lims=True)
    demo_loss_base = float(fresh_net.res_line.pl_mw.sum() + fresh_net.res_trafo.pl_mw.sum())
    
    # 2. AC-OPF
    t_start = time.perf_counter()
    pp.runopp(fresh_net, verbose=False)
    t_opf_demo = (time.perf_counter() - t_start) * 1000.0
    demo_opt_vg = fresh_net.res_gen.vm_pu.values.copy()
    fresh_net.gen.loc[:, 'vm_pu'] = demo_opt_vg
    pp.runpp(fresh_net, enforce_q_lims=True)
    demo_loss_opf = float(fresh_net.res_line.pl_mw.sum() + fresh_net.res_trafo.pl_mw.sum())
    
    # 3. AI Prediction
    demo_u = []
    for l_idx in load_bus_order:
        demo_u.append(fresh_net.load.loc[l_idx, 'p_mw'])
    for l_idx in load_bus_order:
        demo_u.append(fresh_net.load.loc[l_idx, 'q_mvar'])
    demo_u = np.array(demo_u)
    demo_u_scaled = (demo_u - X_mean) / X_std
    
    t_start = time.perf_counter()
    with torch.no_grad():
        demo_ai_vg = model(torch.tensor(demo_u_scaled, dtype=torch.float32).unsqueeze(0)).numpy()[0]
    t_ai_demo = (time.perf_counter() - t_start) * 1000.0
    
    fresh_net.gen.loc[:, 'vm_pu'] = demo_ai_vg
    pp.runpp(fresh_net, enforce_q_lims=True)
    demo_loss_ai = float(fresh_net.res_line.pl_mw.sum() + fresh_net.res_trafo.pl_mw.sum())
    
    print(f"Total Load Active Power:   {fresh_net.load.p_mw.sum():.2f} MW")
    print(f"Total Load Reactive Power: {fresh_net.load.q_mvar.sum():.2f} Mvar\n")
    print("Generator Voltage Setpoints [V_G2, V_G3, V_G6, V_G8]:")
    print(f"  - Nominal Baseline: [1.0000, 1.0000, 1.0000, 1.0000]")
    print(f"  - AC-OPF Optimal:   {demo_opt_vg.round(4)}")
    print(f"  - AI Model Pred:    {demo_ai_vg.round(4)}\n")
    print("Transmission Active Power Losses:")
    print(f"  - Nominal Baseline: {demo_loss_base:.4f} MW")
    print(f"  - AC-OPF Benchmark: {demo_loss_opf:.4f} MW")
    print(f"  - AI Model (Pred):  {demo_loss_ai:.4f} MW")
    print(f"  - Loss Reduction vs Base: {((demo_loss_base - demo_loss_ai)/demo_loss_base)*100:.2f}% (Saved {demo_loss_base - demo_loss_ai:.2f} MW)")
    print(f"  - Optimality Gap vs OPF:  {((demo_loss_ai - demo_loss_opf)/demo_loss_opf)*100:.3f}%\n")
    print(f"Computation Time:")
    print(f"  - AC-OPF:           {t_opf_demo:.2f} ms")
    print(f"  - AI Model:         {t_ai_demo:.4f} ms ({t_opf_demo/t_ai_demo:.1f}x speedup)")
    print("============================================================\n")

if __name__ == "__main__":
    evaluate_all()
