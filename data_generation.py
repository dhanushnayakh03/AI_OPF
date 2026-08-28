import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pandas as pd
import tqdm
import os

def create_and_configure_network(v_min=0.94, v_max=1.10):
    """
    Creates and configures the IEEE-14 bus system for loss-minimizing AC-OPF (ORPD).
    """
    net = nw.case14()
    
    # Set standard IEEE-14 bus voltage operating limits [0.94, 1.10] pu
    net.bus['min_vm_pu'] = v_min
    net.bus['max_vm_pu'] = v_max
    
    # Fix active power generation (P_G) so OPF performs pure Optimal Reactive Power Dispatch
    net.gen['min_p_mw'] = net.gen['p_mw']
    net.gen['max_p_mw'] = net.gen['p_mw']
    net.gen['min_vm_pu'] = v_min
    net.gen['max_vm_pu'] = v_max
    net.gen['controllable'] = True
    
    # Set realistic slack/ext_grid reactive capabilities to avoid artificial slack Q infeasibility
    net.ext_grid['min_q_mvar'] = -100.0
    net.ext_grid['max_q_mvar'] = 100.0
    net.ext_grid['min_p_mw'] = 0.0
    net.ext_grid['max_p_mw'] = 1000.0
    
    # Clear existing cost definitions
    net.poly_cost.drop(net.poly_cost.index, inplace=True)
    net.pwl_cost.drop(net.pwl_cost.index, inplace=True)
    
    # Objective: Minimize external grid active power generation.
    # Since P_loss = P_ext_grid + sum(P_gen) - sum(P_load), and P_gen, P_load are constant per scenario,
    # minimizing P_ext_grid minimizes total transmission loss P_loss exactly.
    pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=1.0)
        
    return net

def generate_data(num_scenarios=1500, save_path="dataset.csv", v_min=0.94, v_max=1.10, seed=42):
    """
    Generates training data by varying load scenarios and solving AC-OPF.
    """
    np.random.seed(seed)
    net = create_and_configure_network(v_min, v_max)
    
    base_p_mw = net.load.p_mw.copy()
    base_q_mvar = net.load.q_mvar.copy()
    num_loads = len(net.load)
    
    # Sort loads by bus index to ensure deterministic feature ordering
    load_bus_order = net.load.sort_values('bus').index.tolist()
    
    dataset = []
    print(f"Generating {num_scenarios} load scenarios for IEEE-14 bus network...")
    
    for _ in tqdm.tqdm(range(num_scenarios)):
        # Independent scaling factor lambda in [0.8, 1.2] for P and Q
        scale_p = np.random.uniform(0.8, 1.2, size=num_loads)
        scale_q = np.random.uniform(0.8, 1.2, size=num_loads)
        
        net.load.loc[:, 'p_mw'] = base_p_mw * scale_p
        net.load.loc[:, 'q_mvar'] = base_q_mvar * scale_q
        
        # Calculate baseline loss under flat nominal 1.0 pu setpoints
        net.gen.loc[:, 'vm_pu'] = [1.0, 1.0, 1.0, 1.0]
        try:
            pp.runpp(net, enforce_q_lims=True)
            loss_flat_base = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
        except Exception:
            continue
        
        # 1. Run AC-OPF to obtain optimal voltage setpoints
        try:
            pp.runopp(net, verbose=False)
        except Exception:
            continue
            
        if not net.OPF_converged:
            continue
            
        # Extract optimal generator voltage setpoints [V_G2, V_G3, V_G6, V_G8]
        optimal_vg = net.res_gen.vm_pu.values.copy()
        
        # 2. Validate with AC Power Flow (pp.runpp)
        net.gen.loc[:, 'vm_pu'] = optimal_vg
        try:
            pp.runpp(net, enforce_q_lims=True)
        except Exception:
            continue
            
        # 3. Check constraints: voltages and line loadings
        v_viol = np.any(net.res_bus.vm_pu < (v_min - 1e-3)) or np.any(net.res_bus.vm_pu > (v_max + 1e-3))
        loading_viol = np.any(net.res_line.loading_percent > 100.0)
        
        if v_viol or loading_viol:
            continue
            
        # Calculate actual physical transmission loss (MW)
        actual_loss = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
        
        # Construct feature row (22 load values: 11 P + 11 Q in bus order)
        row = {}
        for idx, l_idx in enumerate(load_bus_order):
            row[f'P_L{idx+1}'] = net.load.loc[l_idx, 'p_mw']
        for idx, l_idx in enumerate(load_bus_order):
            row[f'Q_L{idx+1}'] = net.load.loc[l_idx, 'q_mvar']
            
        row['V_G2'] = optimal_vg[0]
        row['V_G3'] = optimal_vg[1]
        row['V_G6'] = optimal_vg[2]
        row['V_G8'] = optimal_vg[3]
        row['P_loss_flat_base'] = loss_flat_base
        row['P_loss_OPF'] = actual_loss
        
        dataset.append(row)
        
    df = pd.DataFrame(dataset)
    df.to_csv(save_path, index=False)
    print(f"Successfully generated and verified {len(df)} scenarios. Dataset saved to '{save_path}'.")
    return df

if __name__ == "__main__":
    generate_data(num_scenarios=1500, save_path="dataset.csv")
