import streamlit as st
import gymnasium as gym
import numpy as np
import pandas as pd
import time
import os
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO, DQN, A2C

# =========================================================================
# 1. THE ENVIRONMENT
# =========================================================================

class SmartVentilationEnv(gym.Env):
    MAX_STEPS = 1440
    MAX_PENALTY_COMPONENT = -50.0

    # Preserving your specific hyperparameters
    def __init__(self, alpha=0.4, beta=0.2, gamma=0.25, delta=0.15):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # --- Constants ---
        self.V = 82.72  # Kitchen Volume (m³)

        # --- Pollutant Safe Limits ---
        self.voc_safe = 500.0
        self.pm_safe = 35.0
        self.co2_safe = 2000.0
        self.co2_background = 400.0

        # --- Observation Space ---
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, np.inf, 1.0, 1.0], dtype=np.float32)
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.action_space = Discrete(6)
        self.action_space_map = {
            0: 0.0,
            1: -0.20, 2: -0.10, 3: +0.00, 4: +0.10, 5: +0.20
        }

        # --- Fan Specs ---
        self.fan_low_watt = 40.8
        self.fan_high_watt = 52.9
        self.Q_MAX_ON = 4.2
        self.voc_conc_factor = 4.5
        kitchen_vol = 82.72
        cooking_duration_mins = 13

        voc_factor = (self.voc_conc_factor * kitchen_vol) / cooking_duration_mins
        self.EMISSION_VOC_RATE_MAP = {
            0: 0.0,
            1: 20.0 * voc_factor,  # Air-frying
            2: 30.0 * voc_factor,  # Boiling
            3: 110.0 * voc_factor, # Stir-frying
            4: 230.0 * voc_factor, # Deep-frying
            5: 260.0 * voc_factor  # Pan-frying
        }

        self.EMISSION_PM_RATE_MAP = {
            0: 0.0,
            1: 1.1, 2: 1.1, 3: 178.0, 4: 47.0, 5: 596.0
        }

        self.CO2_EMISSION_MIN = 4.98
        self.CO2_EMISSION_MAX = 6.65

        # --- Init variables ---
        self.voc = 0.0
        self.pm = 0.0
        self.co2 = 400.0
        self.fan_speed = 0.0
        self.current_step = 0
        self.activity_index = 0
        self.activity_list = ["none", "air", "boil", "stir", "deep", "pan"]
        self.steps_remaining_in_activity = 0
        self.meal_tracker = {"breakfast": False, "lunch": False, "dinner": False}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.voc = self.np_random.uniform(10.0, self.voc_safe / 2.0)
        self.pm = self.np_random.uniform(1.0, self.pm_safe / 2.0)
        self.co2 = self.np_random.uniform(400.0, 800.0)
        self.fan_speed = 0.0
        self.activity_index = 0
        self.current_step = 0
        self.steps_remaining_in_activity = 0
        self.meal_tracker = {"breakfast": False, "lunch": False, "dinner": False}
        return self._get_obs(), {"initial_activity": self.activity_list[self.activity_index]}

    def _get_obs(self):
        n_voc = self.voc / self.voc_safe
        n_pm = self.pm / self.pm_safe
        n_co2 = self.co2 / self.co2_safe
        n_act = self.activity_index / 5.0
        current_airflow = self._get_fan_airflow()
        n_flow = current_airflow / self.Q_MAX_ON
        return np.array([n_voc, n_pm, n_co2, n_act, n_flow], dtype=np.float32)

    def _update_fan_speed(self, action: int):
        if action == 0:
            self.fan_speed = 0.0
        else:
            change = self.action_space_map.get(action)
            self.fan_speed = np.clip(self.fan_speed + change, 0.0, 1.0)

    def _get_fan_airflow(self) -> float:
        if self.fan_speed == 0.0: return 0.0
        return np.clip(1.89 * self.fan_speed + 2.31, 2.5, 4.2)

    def _update_pollutants(self):
        base_voc = self.EMISSION_VOC_RATE_MAP.get(self.activity_index, 0.0)
        base_pm = self.EMISSION_PM_RATE_MAP.get(self.activity_index, 0.0)
        if self.activity_index > 0:
            noise_scale = 0.05
            E_VOC = max(0.0, np.random.normal(base_voc, base_voc * noise_scale))
            E_PM = max(0.0, np.random.normal(base_pm, base_pm * noise_scale))
            E_CO2_rate = self.np_random.uniform(self.CO2_EMISSION_MIN, self.CO2_EMISSION_MAX)
        else:
            E_VOC = 0.0; E_PM = 0.0; E_CO2_rate = 0.0

        Q_flow = self._get_fan_airflow()
        removal = Q_flow / self.V
        self.voc += (E_VOC / self.V) - (self.voc * removal)
        self.pm += (E_PM / self.V) - (self.pm * removal)
        self.co2 += E_CO2_rate - ((self.co2 - self.co2_background) * removal)
        self.voc = max(0.0, self.voc); self.pm = max(0.0, self.pm); self.co2 = max(self.co2_background, self.co2)

    def _get_fan_power(self, speed):
        if speed == 0: return 0.0
        return 13.44 * speed + 39.46

    def _reward_func(self, current, limit):
        if current <= limit: return 1.0
        ratio = current / limit
        steepness = 1.0
        penalty = -(np.exp((ratio - 1.0) * steepness) - 1.0)
        return max(self.MAX_PENALTY_COMPONENT, penalty)

    def _calculate_total_reward(self):
        R_voc = self._reward_func(self.voc, self.voc_safe)
        R_pm = self._reward_func(self.pm, self.pm_safe)
        R_co2 = self._reward_func(self.co2, self.co2_safe)
        power = self._get_fan_power(self.fan_speed)
        R_energy = 1.0 if power == 0 else -(power / self.fan_high_watt)
        total = (self.alpha * R_voc + self.beta * R_co2 + self.gamma * R_pm + self.delta * R_energy)
        return {"R_total": total, "R_voc": R_voc, "R_pm": R_pm, "R_co2": R_co2, "R_energy": R_energy}

    def _update_activity(self):
        step = self.current_step
        if self.steps_remaining_in_activity > 0:
            self.steps_remaining_in_activity -= 1
            if self.steps_remaining_in_activity <= 0: self.activity_index = 0
            return self.activity_index
        
        breakfast_start, breakfast_end = 420, 540
        lunch_start, lunch_end = 720, 840
        dinner_start, dinner_end = 1080, 1200

        if breakfast_start <= step < breakfast_end and not self.meal_tracker['breakfast']:
            self._start_random_activity(); self.meal_tracker['breakfast'] = True
        elif lunch_start <= step < lunch_end and not self.meal_tracker['lunch']:
            self._start_random_activity(); self.meal_tracker['lunch'] = True
        elif dinner_start <= step < dinner_end and not self.meal_tracker['dinner']:
            self._start_random_activity(); self.meal_tracker['dinner'] = True
        else:
            self.activity_index = 0
        return self.activity_index

    def _start_random_activity(self):
        self.activity_index = self.np_random.integers(1, 6)
        self.steps_remaining_in_activity = self.np_random.integers(20, 46)

    def step(self, action: int):
        self._update_fan_speed(action)
        self._update_pollutants()
        self.current_step += 1
        self._update_activity()
        
        reward_details = self._calculate_total_reward()
        reward = reward_details["R_total"]
        observation = self._get_obs()
        terminated = False
        truncated = self.current_step >= self.MAX_STEPS

        # --- THIS IS THE FIX ---
        info = {
            "activity": self.activity_list[self.activity_index],
            "fan_power": self._get_fan_power(self.fan_speed),
            "fan_speed_ratio": self.fan_speed,  # <--- Added this so the gauge works!
            "reward_details": reward_details,
            "voc": self.voc,
            "pm": self.pm,
            "co2": self.co2
        }
        
        return observation, reward, terminated, truncated, info

# =========================================================================
# 2. THE DASHBOARD UI
# =========================================================================

st.set_page_config(page_title="Smart Ventilation System", layout="wide", page_icon="🌬️")

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #000000; }
    
    div[data-testid="stMetric"] {
        background-color: #FFFFFF; border: 1px solid #E0E0E0;
        padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    canvas { min-height: 200px !important; } 

    /* 🔥 ANIMATIONS */
    @keyframes spin { 
        from { transform: rotate(0deg); } 
        to { transform: rotate(360deg); } 
    }
    
    /* Spinning State */
    .fan-spin {
        animation: spin 1s linear infinite; /* Smooth continuous spin */
        width: 100px; 
        height: 100px;
        display: block; 
        margin: 0 auto;
    }
    
    /* Off State (Static & Grey) */
    .fan-off {
        width: 100px; 
        height: 100px;
        display: block; 
        margin: 0 auto;
        opacity: 0.3;
        filter: grayscale(100%);
    }
    
    /* Cooking Animation */
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .cook-anim {
        animation: bounce 0.6s infinite alternate;
        font-size: 80px; display: block; margin: 0 auto; text-align: center;
    }
    .cook-idle {
        font-size: 80px; opacity: 0.2; display: block; margin: 0 auto; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍳 Smart Ventilation ")

# ---------------------------------------------------------------------
# 📂 SMART SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.header("🎛️ Control Room")

# 1. Define folder path
models_dir = "models"
if not os.path.exists(models_dir): os.makedirs(models_dir)

model_files = []
for root, dirs, files in os.walk(models_dir):
    for file in files:
        if file.endswith(".zip"):
            rel_path = os.path.relpath(os.path.join(root, file), models_dir)
            model_files.append(rel_path)

model_path = None
if model_files:
    selected_file = st.sidebar.selectbox("📂 Select Saved Model", model_files)
    model_path = os.path.join(models_dir, selected_file)
else:
    st.sidebar.info("No models found in 'models/' folder.")
    uploaded = st.sidebar.file_uploader("Upload Model (.zip)", type="zip")
    if uploaded:
        with open("temp_model.zip", "wb") as f: f.write(uploaded.getbuffer())
        model_path = "temp_model.zip"

# 🔥 THE CLEANEST LOADING LOGIC 🔥
if model_path:
    # Only load if the user picked a different file
    if "current_model_path" not in st.session_state or st.session_state.current_model_path != model_path:
        
        loaded_model = None
        
        # We just loop through the classes. 
        # Note: DQN loader works for both DQN and DDQN files.
        for algo_class in [PPO, DQN, A2C]:
            try:
                loaded_model = algo_class.load(model_path, env=None)
                break # Success! Stop trying others.
            except:
                continue # Failed, try next one.

        if loaded_model:
            st.session_state.model = loaded_model
            st.session_state.current_model_path = model_path
            
            # Reset simulation because we have a new brain
            if "env" in st.session_state:
                st.session_state.playing = False
                st.session_state.day_results = []
        else:
            st.session_state.model = None
            st.sidebar.error("❌ Error: Model format not recognized.")
            
# ---------------------------------------------------------------------
st.sidebar.subheader("⏩ Turbo Settings")
turbo_speed = st.sidebar.slider("Idle Turbo Speed (Delay)", 0.05, 1.0, 0.2)

st.sidebar.markdown("---")
st.sidebar.subheader("🗓️ Simulation Config")
# 🔥 We check this value later to trigger auto-reset
num_days = st.sidebar.slider("Evaluation Days", 1, 30, 1)

with st.sidebar.expander("Advanced Physics Params"):
    p_alpha = st.number_input("Alpha (VOC)", 0.0, 1.0, 0.4)
    p_beta = st.number_input("Beta (CO2)", 0.0, 1.0, 0.2)
    p_gamma = st.number_input("Gamma (PM)", 0.0, 1.0, 0.25)
    p_delta = st.number_input("Delta (Energy)", 0.0, 1.0, 0.15)

# =========================================================================
# 🔄 CENTRAL RESET FUNCTION (The Fix)
# =========================================================================
def reset_simulation():
    # 1. Re-init Env
    st.session_state.env = SmartVentilationEnv(alpha=p_alpha, beta=p_beta, gamma=p_gamma, delta=p_delta)
    
    # 2. Reset Observation
    obs, _ = st.session_state.env.reset()
    st.session_state.obs = obs
    
    # 3. Reset Variables
    st.session_state.playing = False
    st.session_state.current_day = 1
    st.session_state.day_results = []
    
    # 4. Reset Log
    initial_info = {
        "activity": "none", "fan_power": 0.0, "fan_speed_ratio": 0.0,
        "reward_details": st.session_state.env._calculate_total_reward(),
        "voc": 0.0, "pm": 0.0, "co2": 400.0
    }
    st.session_state.log = [initial_info]

# --- SESSION STATE INITIALIZATION ---
if "env" not in st.session_state:
    # Initialize first time
    reset_simulation() 
    # Store initial num_days to detect changes
    st.session_state.saved_num_days = num_days 

if "model" not in st.session_state: st.session_state.model = None

# 🔥 AUTO-RESET: Detect if user moved the slider
if "saved_num_days" in st.session_state and st.session_state.saved_num_days != num_days:
    st.session_state.saved_num_days = num_days # Update memory
    reset_simulation() # Force clean slate
    st.rerun() # Refresh UI immediately to show "Day 1/1"

# --- CONTROLS ---
c1, c2, c3, c4 = st.columns([1, 1, 1, 4])
if c1.button("▶ PLAY"): st.session_state.playing = True
if c2.button("⏸ PAUSE"): st.session_state.playing = False

# 🔄 RESET BUTTON (Now uses the shared function)
if c3.button("🔁 RESET"):
    reset_simulation()
    st.rerun()

# =========================================================================
# 🏗️ SKELETON SETUP
# =========================================================================
status_spot = st.empty() 
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1: m1_spot = st.empty()
with col2: m2_spot = st.empty()
with col3: m3_spot = st.empty()
with col4: m4_spot = st.empty()

st.markdown("---")
k_col, f_col = st.columns(2)
with k_col: k_spot = st.empty()
with f_col: f_spot = st.empty()

st.markdown("---")
c_voc, c_pm, c_co2 = st.columns(3)
with c_voc:
    st.markdown("#### 🌫️ VOC (ppb)")
    voc_chart_spot = st.empty()
with c_pm:
    st.markdown("#### 🌬️ PM2.5 (µg/m³)")
    pm_chart_spot = st.empty()
with c_co2:
    st.markdown("#### 🌡️ CO2 (ppm)")
    co2_chart_spot = st.empty()

# =========================================================================
# 🛠️ HELPER: DRAW CHARTS
# =========================================================================
def draw_charts():
    full_df = pd.DataFrame(st.session_state.log)
    voc_chart_spot.line_chart(full_df[["voc"]], height=200, color="#FF4B4B")
    pm_chart_spot.line_chart(full_df[["pm"]], height=200, color="#1C83E1")
    co2_chart_spot.line_chart(full_df[["co2"]], height=200, color="#00C0F2")

# =========================================================================
# 🛠️ HELPER: UPDATE DASHBOARD
# =========================================================================
def update_dashboard(is_busy_mode=False):
    last = st.session_state.log[-1]
    current_step = len(st.session_state.log) - 1
    
    day_str = f"Day {st.session_state.current_day}/{num_days}"
    if is_busy_mode:
        status_spot.markdown(f"**🐢 ACTIVE MODE** | {day_str} | Step: `{current_step}`")
    else:
        status_spot.markdown(f"**🚀 TURBO IDLE** | {day_str} | Step: `{current_step}`")
    
    m1_spot.metric("⏱️ Time", f"{current_step}")
    
    voc_val = last['voc']
    m2_spot.metric("🌫️ VOC (ppb)", f"{voc_val:.1f}", 
                   delta="High" if voc_val > 500 else None, delta_color="inverse")
    
    pm_val = last['pm']
    m3_spot.metric("🌬️ PM2.5", f"{pm_val:.1f}", 
                   delta="High" if pm_val > 35 else None, delta_color="inverse")
    
    m4_spot.metric("🌡️ CO2", f"{last['co2']:.0f}")

    # 3. KITCHEN VISUALS
    with k_spot.container():
        act = last['activity']
        if act != "none":
            st.markdown(f"<div class='cook-anim'>🍳</div>", unsafe_allow_html=True)
            st.error(f"🔥 ACTIVITY: {act.upper()}")
        elif (last['voc'] > 500):
            st.markdown(f"<div class='cook-idle'>🌫️</div>", unsafe_allow_html=True)
            st.warning("⚠️ CLEARING FUMES")
        else:
            st.markdown(f"<div class='cook-idle'>💤</div>", unsafe_allow_html=True)
            st.success("✅ IDLE & SAFE")

    # 4. FAN VISUALS (Dynamic SVG Icon)
    with f_spot.container():
        fr = last.get('fan_speed_ratio', 0.0)
        
        # Custom 3-Blade Fan SVG
        fan_svg = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 640">
        <!--!Font Awesome Free v7.1.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2025 Fonticons, Inc.-->
        <path fill="#000000" d="M224 208C224 128.5 288.5 64 368 64C376.8 64 384 71.2 384 80L384 232.2C399 226.9 415.2 224 432 224C511.5 224 576 288.5 576 368C576 376.8 
        568.8 384 560 384L407.8 384C413.1 399 416 415.2 416 432C416 511.5 351.5 576 272 576C263.2 576 256 568.8 256 560L256 407.8C241 413.1 224.8 416 208 416C128.5 416 64 351.5 
        64 272C64 263.2 71.2 256 80 256L232.2 256C226.9 241 224 224.8 224 208zM320 352C337.7 352 352 337.7 352 320C352 302.3 337.7 288 320 288C302.3 288 288 302.3 288 320C288 
        337.7 302.3 352 320 352z"/></svg>
        """
        
        if fr > 0.01:
            # If speed is 1.0 (Max) -> Duration 0.2s (Fast spin)
            # If speed is 0.1 (Min) -> Duration 1.5s (Slow spin)
            anim_duration = max(0.2, 1.5 - (fr * 1.3))
            
            # Inject duration directly into style
            st.markdown(f"<div class='fan-spin' style='animation-duration: {anim_duration:.2f}s;'>{fan_svg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='fan-off'>{fan_svg}</div>", unsafe_allow_html=True)
            
        st.progress(fr)
        st.caption(f"Speed: {int(fr*100)}% | Power: {last['fan_power']:.1f} W")

    draw_charts()

update_dashboard(is_busy_mode=False)

# =========================================================================
# 🔄 SMART LOOP
# =========================================================================
while st.session_state.playing and st.session_state.model:
    
    # 0. Safety Check: If config changed mid-run to something impossible
    if st.session_state.current_day > num_days:
        st.session_state.playing = False
        st.warning("⚠️ Simulation stopped: Days config changed.")
        break

    last = st.session_state.log[-1]
    
    cooking = (last['activity'] != "none")
    fan_on = (last.get('fan_speed_ratio', 0) > 0.01)
    
    is_busy = False
    if cooking or fan_on: is_busy = True
    
    if is_busy:
        batch_size = 1
        current_delay = 0.5 
    else:
        batch_size = 10 
        current_delay = turbo_speed

    for _ in range(batch_size):
        action, _ = st.session_state.model.predict(st.session_state.obs, deterministic=True)
        obs, reward, terminated, truncated, info = st.session_state.env.step(int(action))
        
        st.session_state.obs = obs
        st.session_state.log.append(info)
        
        if batch_size > 1:
            step_busy = (info['activity'] != "none") or \
                        (info.get('fan_speed_ratio', 0) > 0.01) or \
                        (info['voc'] > 500)
            if step_busy:
                is_busy = True 
                break 
        
        # --- MULTI-DAY LOGIC ---
        if terminated or truncated:
            day_stats = {
                "Day": st.session_state.current_day,
                "Avg VOC": np.mean([x['voc'] for x in st.session_state.log]),
                "Avg PM": np.mean([x['pm'] for x in st.session_state.log]),
                "Avg CO2": np.mean([x['co2'] for x in st.session_state.log]),
                "Unsafe Steps": sum(1 for x in st.session_state.log if x['voc'] > 500)
            }
            st.session_state.day_results.append(day_stats)
            
            if st.session_state.current_day < num_days:
                st.session_state.current_day += 1
                st.session_state.obs, _ = st.session_state.env.reset()
                st.session_state.log = [st.session_state.log[-1]]
                st.session_state.log = [{
                    "activity": "none", "fan_power": 0.0, "fan_speed_ratio": 0.0,
                    "reward_details": st.session_state.env._calculate_total_reward(),
                    "voc": 0.0, "pm": 0.0, "co2": 400.0
                }]
                break 
            else:
                st.session_state.playing = False
                st.success("✅ Simulation Complete!")
                break
    
    update_dashboard(is_busy_mode=is_busy)
    if not st.session_state.playing: break
    time.sleep(current_delay)

# =========================================================================
# 📊 FINAL RESULTS
# =========================================================================
if not st.session_state.playing and st.session_state.day_results:
    st.divider()
    st.header("📊 Final Campaign Report")
    
    df_res = pd.DataFrame(st.session_state.day_results)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Days", f"{len(df_res)}")
    c2.metric("Avg Unsafe Steps/Day", f"{df_res['Unsafe Steps'].mean():.1f}", delta_color="inverse")
    c3.metric("Avg VOC Level", f"{df_res['Avg VOC'].mean():.1f} ppb")
    
