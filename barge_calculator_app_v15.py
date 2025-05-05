# barge_calculator_app_v15.py
# Streamlit App with High-Precision 2D Integration for Buoyancy and Center of Buoyancy

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Page Config and Style ---
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-size: 12px !important;
            line-height: 1.2em !important;
        }
        .stSlider label, .stNumberInput label, .stTextInput label {
            font-size: 11px !important;
        }
        .stSlider div[data-baseweb="slider"] {
            padding: 0.2em 0 !important;
        }
        .stTextInput input, .stNumberInput input {
            font-size: 11px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
g = 9.8
rho_w_default = 1000.0

# --- Function: 2D Integration for Buoyancy and B ---
def calculate_buoyancy_and_B(X1, X2, Y1, Y2, H, h, theta_x, theta_y, rho_w, g=9.8, Nx=100, Ny=100):
    x_vals = np.linspace(-X2, X1, Nx)
    y_vals = np.linspace(-Y2, Y1, Ny)
    dx = (X1 + X2) / Nx
    dy = (Y1 + Y2) / Ny
    X, Y = np.meshgrid(x_vals, y_vals)
    depth = h - X * np.tan(theta_y) - Y * np.tan(theta_x)
    depth_clipped = np.clip(depth, 0, H)
    dV = depth_clipped * dx * dy
    volume = np.sum(dV)
    FB = volume * rho_w * g
    Bx = np.sum(dV * X) / volume
    By = np.sum(dV * Y) / volume
    KB = np.sum(dV * (depth_clipped / 2)) / volume
    return {"FB": FB, "Volume": volume, "Bx": Bx, "By": By, "KB": KB}

# --- App Title ---
st.title("Barge Calculator v13 – 2D Integration Enhanced")

# --- Barge Parameters ---
col4, col5, col6, col7, col8 = st.columns(5)
with col4:
    X1 = st.number_input("Length Forward (X1) [m]", value=5.0)
with col5:
    X2 = st.number_input("Length Aft (X2) [m]", value=5.0)
with col6:
    Y1 = st.number_input("Width Port (Y1) [m]", value=3.0)
with col7:
    Y2 = st.number_input("Width Starboard (Y2) [m]", value=3.0)
with col8:
    H = st.number_input("Barge Height (H) [m]", value=2.0)

col1, col2, col3 = st.columns(3)
with col1:
    W_b = st.number_input("Barge Self-weight [kg]", value=10000.0)
with col2:
    rho_w = st.number_input("Water Density [kg/m^3]", value=rho_w_default)
with col3:
    Zb = st.slider("Barge CG Height Zb [m]", 0.0, H, H / 2, 0.01)

# --- Equipment Input ---
default_data = {
    'Name': ["P+M No.1", "P+M No.2", "Panel", "Valve No.1", "Valve No.2", "Header pipe", "Worker"],
    'Weight [kg]': [4500, 4500, 600, 500, 500, 1500, 70],
    'X [m]': [1.5, -1.5, 0.0, 1.5, -1.5, 0.0, 2.0],
    'Y [m]': [0.0, 0.0, -1.5, 1.0, 1.0, 2.5, 2.0],
    'Z [m]': [2.0, 2.0, 1.3, 0.5, 0.5, 1.0, 2.0]
}
equipment_list = []
equip_n = st.number_input("Number of Equipment", 1, 20, len(default_data['Name']))
for i in range(equip_n):
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            name = st.text_input(f"Name {i+1}", default_data['Name'][i] if i < len(default_data['Name']) else f"Eq{i+1}", key=f"name_{i}")
            weight = st.slider(f"Weight [kg] - {name}", 0, 5000, default_data['Weight [kg]'][i] if i < len(default_data['Weight [kg]']) else 100, 10, key=f"weight_{i}")
        with col2:
            x = st.slider(f"X [m] - {name}", -5.0, 5.0, default_data['X [m]'][i] if i < len(default_data['X [m]']) else 0.0, 0.1, key=f"x_{i}")
        with col3:
            y = st.slider(f"Y [m] - {name}", -5.0, 5.0, default_data['Y [m]'][i] if i < len(default_data['Y [m]']) else 0.0, 0.1, key=f"y_{i}")
        with col4:
            z = st.slider(f"Z [m] - {name}", 0.0, 4.0, default_data['Z [m]'][i] if i < len(default_data['Z [m]']) else 1.0, 0.1, key=f"z_{i}")
        equipment_list.append({'Name': name, 'Weight [kg]': weight, 'X [m]': x, 'Y [m]': y, 'Z [m]': z})
equipment_df = pd.DataFrame(equipment_list)

# --- Geometry and Basic Stability ---
L = X1 + X2
B = Y1 + Y2
Xb = (X1 - X2) / 2
Yb = (Y1 - Y2) / 2
W_load = equipment_df['Weight [kg]'].sum()
W_total = W_b + W_load

xg = (equipment_df['Weight [kg]'] * equipment_df['X [m]']).sum() + W_b * Xb
yg = (equipment_df['Weight [kg]'] * equipment_df['Y [m]']).sum() + W_b * Yb
zg = (equipment_df['Weight [kg]'] * equipment_df['Z [m]']).sum() + W_b * Zb
xg /= W_total
yg /= W_total
zg /= W_total
h = W_total / (rho_w * L * B)
volume_disp = L * B * h
FB = volume_disp * rho_w
KB = h / 2
KG = zg

# --- GM and Initial Tilt ---
Ixx = (B * h**3) / 12
Iyy = (L * h**3) / 12
BMx = Ixx / volume_disp
BMy = Iyy / volume_disp
GMx = KB + BMx - KG
GMy = KB + BMy - KG
Mx = (equipment_df['Weight [kg]'] * equipment_df['Y [m]']).sum() + W_b * Yb
My = (equipment_df['Weight [kg]'] * equipment_df['X [m]']).sum() + W_b * Xb
theta_x = np.arcsin(np.clip(Mx / (FB * GMx), -1.0, 1.0))
theta_y = np.arcsin(np.clip(My / (FB * GMy), -1.0, 1.0))

# --- 2D Integration (Accurate Buoyancy and B) ---
result_2d = calculate_buoyancy_and_B(X1, X2, Y1, Y2, H, h, theta_x, theta_y, rho_w)
FB_2d = result_2d["FB"]
volume_disp_2d = result_2d["Volume"]
Bx_2d = result_2d["Bx"]
By_2d = result_2d["By"]
KB_2d = result_2d["KB"]

# --- GZ and Tilt Re-evaluation ---
GZx_2d = yg - By_2d
GZy_2d = xg - Bx_2d
theta_x_2d = np.arcsin(np.clip(GZx_2d / GMx, -1.0, 1.0))
theta_y_2d = np.arcsin(np.clip(GZy_2d / GMy, -1.0, 1.0))

# --- Output Summary with 3 Columns ---
st.subheader("Summary with 2D Integration")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"Total Weight: {W_total:.2f} kg")
    st.write(f"Draft: {h:.3f} m")
    st.write(f"Displaced Volume: {volume_disp_2d:.3f} m³")
with col2:
    st.write(f"CG (X, Y, Z): ({xg:.2f}, {yg:.2f}, {zg:.2f})")
    st.write(f"Center of Buoyancy (Bx, By): ({Bx_2d:.2f}, {By_2d:.2f})")
    st.write(f"KB (Z of B): {KB_2d:.3f} m")
with col3:
    st.write(f"GMx: {GMx:.3f} m, GMy: {GMy:.3f} m")
    st.write(f"Heel Y: {np.degrees(theta_x_2d):.2f}°, Trim X: {np.degrees(theta_y_2d):.2f}°")

# --- CG and Tilt Plot (X-Z / Y-Z Views) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# X-Z View
ax1.set_title("X-Z View")
ax1.plot([-X2, X1], [x * np.tan(theta_y_2d) for x in [-X2, X1]], 'k--')
ax1.plot([-X2, X1], [x * np.tan(theta_y_2d) + H for x in [-X2, X1]], 'k--')
ax1.plot(xg, zg, 'ro', label='CG')
ax1.plot(Bx_2d, KB_2d, 'c^', label='B (shifted)')
ax1.plot(Xb, Zb, 'bx', label='Barge CG')
for _, r in equipment_df.iterrows():
    ax1.plot(r['X [m]'], r['Z [m]'], 'gs')
    ax1.text(r['X [m]'], r['Z [m]'] + 0.1, r['Name'], fontsize=7, color='green', ha='center')
ax1.axhline(h, color='blue', linestyle='--', label='Waterline')
ax1.set_xlim(-X2 - 0.5, X1 + 0.5)
ax1.set_ylim(0, H + 0.5)
ax1.set_xlabel("X [m]")
ax1.set_ylabel("Z [m]")
ax1.grid()
ax1.legend()

# Y-Z View
ax2.set_title("Y-Z View")
ax2.plot([-Y2, Y1], [y * np.tan(theta_x_2d) for y in [-Y2, Y1]], 'k--')
ax2.plot([-Y2, Y1], [y * np.tan(theta_x_2d) + H for y in [-Y2, Y1]], 'k--')
ax2.plot(yg, zg, 'ro', label='CG')
ax2.plot(By_2d, KB_2d, 'c^', label='B (shifted)')
ax2.plot(Yb, Zb, 'bx', label='Barge CG')
for _, r in equipment_df.iterrows():
    ax2.plot(r['Y [m]'], r['Z [m]'], 'gs')
    ax2.text(r['Y [m]'], r['Z [m]'] + 0.1, r['Name'], fontsize=7, color='green', ha='center')
ax2.axhline(h, color='blue', linestyle='--', label='Waterline')
ax2.set_xlim(-Y2 - 0.5, Y1 + 0.5)
ax2.set_ylim(0, H + 0.5)
ax2.set_xlabel("Y [m]")
ax2.set_ylabel("Z [m]")
ax2.grid()
ax2.legend()

st.pyplot(fig)

# GZ and moment curve
heel_angles = np.linspace(-20, 20, 200)
GZx_list, GZy_list, Mx_list, My_list = [], [], [], []
for deg in heel_angles:
    t = np.radians(deg)
    h_eff = h * np.cos(t)
    Vx = L * (B * np.cos(t)) * h_eff
    Ixx = (B * np.cos(t)) * h_eff**3 / 12
    GZx = (h_eff/2 + Ixx/Vx - KG) * np.sin(t)
    GZx_list.append(GZx)
    Mx_list.append(GZx * Vx * rho_w)
    Vy = (L * np.cos(t)) * B * h_eff
    Iyy = (L * np.cos(t)) * h_eff**3 / 12
    GZy = (h_eff/2 + Iyy/Vy - KG) * np.sin(t)
    GZy_list.append(GZy)
    My_list.append(GZy * Vy * rho_w)

fig2, ax = plt.subplots(figsize=(8, 5))
ax.plot(heel_angles, GZx_list, label='GZx [m]')
ax.plot(heel_angles, GZy_list, label='GZy [m]', linestyle='--')
ax.plot(heel_angles, Mx_list, label='Mx [kgf·m]', linestyle=':')
ax.plot(heel_angles, My_list, label='My [kgf·m]', linestyle=':')
ax.axhline(0, color='gray', linestyle=':')
ax.set_xlabel("Tilt Angle [deg]")
ax.set_ylabel("GZ / Moment")
ax.set_title("GZ and Restoring Moment")
ax.grid()
ax.legend()
st.pyplot(fig2)

# --- Safety Check Function ---
def check_safety(X1, X2, Y1, Y2, xg, yg, GMx, GMy, theta_x_deg, theta_y_deg, equipment_df):
    warnings = []
    # GMの符号チェック
    if GMx < 0:
        warnings.append("❌ GMx < 0: The barge is unstable in X-direction (may capsize).")
    if GMy < 0:
        warnings.append("❌ GMy < 0: The barge is unstable in Y-direction (may capsize).")
    # 傾き角の過大チェック
    if abs(theta_x_deg) > 30:
        warnings.append(f"⚠️ Heel angle X exceeds 30°: {theta_x_deg:.2f}°")
    if abs(theta_y_deg) > 30:
        warnings.append(f"⚠️ Trim angle Y exceeds 30°: {theta_y_deg:.2f}°")
    # 重心のデッキ内チェック
    if not (-X2 <= xg <= X1):
        warnings.append("❌ CG X is outside the deck")
    if not (-Y2 <= yg <= Y1):
        warnings.append("❌ CG Y is outside the deck")
    # 各装備品の位置確認
    for _, row in equipment_df.iterrows():
        if not (-X2 <= row['X [m]'] <= X1):
            warnings.append(f"❌ {row['Name']} is outside X deck")
        if not (-Y2 <= row['Y [m]'] <= Y1):
            warnings.append(f"❌ {row['Name']} is outside Y deck")
    return warnings

st.subheader("🛑 Safety Warnings")
warnings = check_safety(
    X1, X2, Y1, Y2,
    xg=xg, yg=yg,
    GMx=GMx, GMy=GMy,
    theta_x_deg=np.degrees(theta_x_2d),
    theta_y_deg=np.degrees(theta_y_2d),
    equipment_df=equipment_df
)

unstable = (GMx < 0 or GMy < 0 or abs(np.degrees(theta_x_2d)) > 30 or abs(np.degrees(theta_y_2d)) > 30)

if warnings:
    st.warning(f"⚠️ {len(warnings)} warning(s) found:")
    for w in warnings:
        st.error(w)

if unstable:
    st.markdown("### ❗️ **Unstable configuration detected. Please adjust loading or CG height.**", unsafe_allow_html=True)
elif not warnings:
    st.success("✅ All equipment and CG are within deck bounds and GM is positive")

# python -m streamlit run barge_calculator_app_v15.py