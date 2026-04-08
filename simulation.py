import numpy as np
import bisect
import time


def make_coe_array(array):
    half_term = ((array[0:-1] + array[1:])/2.0)
    return np.append(half_term, array[-1])
def cp(array):
    plt.figure()
    plt.plot(array)
    plt.show()
def make_x_array(x, sub_channel):
    res_x = list(x)
    res_y = [[0,0,0,0] for _ in range(len(res_x))]    
    sub_channel_sorted = sorted(sub_channel, key=lambda item: item[0])
    for point_x, point_y in sub_channel_sorted:
        count = res_x.count(point_x)
        if count == 0:
            idx = bisect.bisect_left(res_x, point_x)
            res_x.insert(idx, point_x)
            res_y.insert(idx, list(point_y)) 
            res_x.insert(idx + 1, point_x)
            res_y.insert(idx + 1, [0,0,0,0])    
        elif count == 1:
            idx = bisect.bisect_left(res_x, point_x)
            res_x.insert(idx, point_x)
            res_y.insert(idx, list(point_y))
    return res_x, res_y
def clean_x_array(res_x, res_y1, res_y2):
    clean_x = []
    clean_y1 = []
    clean_y2 = []
    n = len(res_x)
    for i in range(n):
        if i < n - 1 and res_x[i] == res_x[i+1]:
            continue
        clean_x.append(res_x[i])
        clean_y1.append(res_y1[i])
        clean_y2.append(res_y2[i])
    return clean_x, clean_y1, clean_y2



class SaintVenantSolver:
    def __init__(self, theta, g, alpha, n, n_sections, x, t, dx, dt):
        self.N = n_sections
        self.x = x
        self.t = t
        
        # self.dx = dx
        self.dx = ((self.x[1:] - self.x[0:-1]))
        self.dx = np.append(self.dx, self.dx[-1])
        self.dt = dt

        
        # 状态变量：当前时刻 (n) 和 下一时刻 (n+1)
        self.Z = np.zeros(n_sections)
        self.Q = np.zeros(n_sections)
        self.Z_next = np.zeros(n_sections)
        self.Q_next = np.zeros(n_sections)

        self.sub_channel = []
        self.Qf = []

        # 方程系数
        self.theta = theta
        self.g = g
        self.alpha = alpha
        self.n = n
        self.B = np.zeros(n_sections)
        self.R = np.zeros(n_sections)
        self.A = np.zeros(n_sections)
        self.u = np.zeros(n_sections)
        self.q = np.zeros(n_sections)
        self.C = np.zeros(n_sections)

        
        # 追赶系数 
        self.S = np.zeros(n_sections)
        self.T = np.zeros(n_sections)
        self.P = np.zeros(n_sections)
        self.V = np.zeros(n_sections)

    def initialize_coe(self):
        self.B = np.zeros(self.N)
        self.R = np.zeros(self.N)
        self.A = np.zeros(self.N)
        self.u = np.zeros(self.N)
        self.q = np.zeros(self.N)
        self.C = np.zeros(self.N)

        
        # 追赶系数 
        self.S = np.zeros(self.N)
        self.T = np.zeros(self.N)
        self.P = np.zeros(self.N)
        self.V = np.zeros(self.N)

        self.Qf = []
        

    def set_initial_conditions(self, Z0, Q0):
        """设置初始水位和流量"""
        self.Z = np.array(Z0)
        self.Q = np.array(Q0)
        self.Z_next = self.Z.copy()
        self.Q_next = self.Q.copy()

    def set_boundary_condition(self, BC_up, BC_down):
        # if BC_up == "Z_up":
        #     self.Z[0] = boundary_up
        # else:
        #     self.Q[0] = boundary_up

        # if BC_down == "Q_down":
        #     self.Q[-1] = boundary_down
        # else:
        #     self.Z[-1] = boundary_down
        self.BC_up = BC_up
        self.BC_down = BC_down

    # def set_coefficients(self, B_array, R_array, A_array, C_array, u_array, q_array):
    #     # At time n
    #     self.B = b_array
    #     self.R = R_array
    #     self.A = A_array
    #     self.u = u_array
    #     self.q = q_array
    #     self.C = C_array

    def set_sub_channel(self, sub_channel):
        
        # pos_array = sub_channel[:, 0]
        # indices = np.array([np.abs(self.x - p).argmin() for p in pos_array])
        # sub_channel[:, 0] = indices
        ### data format example:
        # x : [0, 1, 2, 2.5, 2.5, 3, 4, 5]
        # sc : [0, 0, 0, qf, 0, 0, 0, 0]
        self.sub_channel = sub_channel
        # self.Qf = np.array([0]*len(self.sub_channel))

    def update_Qf(self):
        # fsc = np.array([x for x in self.sub_channel if len(x) > 0 and x[0] != 0], dtype=object)
        mask = np.array([item[0] != 0 for item in self.sub_channel])
        fsc = self.sub_channel[mask]
        indices = np.where(mask)[0]
        for ii in range(len(fsc)):
            sigma_s = fsc[ii][0]
            b = fsc[ii][1]
            e = fsc[ii][2]
            Zd = fsc[ii][3]

            A_coe = sigma_s*b*e
            z = self.Z[indices[ii]]

            qf = A_coe * np.sqrt(2.0*self.g*(z))
            (self.Qf).append(qf)
            

    def set_coefficients(self, Z, Q, q):
        # 矩形渠道

        camber = np.pi/12
        slop = 1.0/1500.0
        channel_length = 1000
        d = 1.16
        
        h = Z - ((channel_length - self.x)*slop)

        self.B = 2.0*h*np.tan(camber) + d
        P = d + 2.0*h/np.cos(camber)
        self.A = (self.B + d)*h/2.0
        self.R = self.A / P
        self.u = Q/self.A
        self.q = q
        self.C = 1.0/self.n * np.power(self.R, 1/6)

    def get_discrete_coeffs(self, j):
        """
        计算离散方程系数 Cj, Fj, Ej, Gj, Dj, Phij 
        注：此处应根据具体的差分格式（如Preissmann）和断面几何计算
        """
        # 这里仅为示例占位符，实际需填入几何计算逻辑

        # All of these coefficients may need to drop the last value

        
        B_term = make_coe_array(self.B)
        Cj = B_term*self.dx / (2.0*self.dt*self.theta)
        
        q_term = make_coe_array(self.q)
        Q_diff = np.append(((self.Q[1:] - self.Q[0:-1])), self.Q[-1])
        Z_sum = np.append(((self.Z[1:] + self.Z[0:-1])), self.Z[-1])
        Dj = q_term*self.dx/self.theta - (1-self.theta)/self.theta*(Q_diff) + Cj*(Z_sum)
        
        g_term = (self.g*np.abs(self.u)/2.0/self.theta/self.C/self.C/self.R)
        g_term_adv = np.append(g_term[1:], g_term[-1])
        u_term_adv = np.append(self.u[1:], self.u[-1])
        Ej = self.dx/(2.0*self.theta*self.dt) - (self.alpha*self.u) + g_term_adv*self.dx
        Gj = self.dx/(2.0*self.theta*self.dt) + (self.alpha*u_term_adv) + g_term_adv*self.dx
        
        A_term = make_coe_array(self.A)
        Fj = self.g*A_term

        Q_sum = np.append(((self.Q[1:] + self.Q[0:-1])), self.Q[-1])
        Z_diff = np.append(((self.Z[1:] - self.Z[0:-1])), self.Z[-1])
        auQ = self.alpha*self.u*self.Q
        auQ_diff = np.append(((auQ[1:] - auQ[0:-1])), auQ[-1])
        Phij = self.dx/(2.0*self.theta*self.dt)*Q_sum - (1.0-self.theta)/self.theta*(auQ_diff+Fj*Z_diff)
        
        return Cj, Fj, Ej, Gj, Dj, Phij

    def get_catchup_coefficient(self, Cj, Fj, Ej, Gj, Dj, Phij, BC, j):
        if BC == "Z_up":
            # 计算中间变量 Y
            Y1 = Dj - Cj * self.P[j]
            Y2 = Phij + Fj * self.P[j]
            Y3 = 1 + Cj * self.V[j]
            Y4 = Ej + Fj * self.V[j]
            
            # 计算下一断面的追赶系数
            denom = Fj * Y3 + Cj * Y4 # 分母项
            self.S[j+1] = (Cj * Y2 - Fj * Y1) / denom
            self.T[j+1] = (Cj * Gj - Fj) / denom
            
            # 计算 Pj+1 和 Vj+1
            # 这里的推导依赖于 Zj+1 = Pj+1 - Vj+1 * Qj+1 
            # 具体的 P, V 表达式需根据式(4-58)进一步代数整理
            self.P[j+1] = (Y1 + Y3 * self.S[j+1]) / Cj 
            self.V[j+1] = (Y3 * self.T[j+1] - 1) / Cj 
        elif BC == "Q_up":

            if self.dx[j] == 0:
                if self.sub_channel[j][0] != 0:
                    # print("Hiting sub-channel!")

                    sigma_s = self.sub_channel[j][0]
                    b = self.sub_channel[j][1]
                    e = self.sub_channel[j][2]
                    Zd = self.sub_channel[j][3]

                    A_coe = -sigma_s*b*e
                    delta_Z = self.Z[j] - Zd

                    

                    Qf = A_coe * np.sqrt(2.0*self.g*(self.Z[j]))
                    p2 = -self.g*A_coe*np.sqrt(2.0*self.g*delta_Z)*self.Z[j]
                    v2 = -self.g*A_coe*np.sqrt(2.0*self.g*delta_Z)
                    
                    self.S[j+1] = 0
                    self.T[j+1] = -1
                    self.P[j+1] = self.P[j] + Qf + p2
                    self.V[j+1] = self.V[j] + v2

                    # print("P[j]: " + str(self.P[j]))
                    # print("Qf: " + str(Qf))
                    # print("p2: " + str(p2))
                    # print("V[j]: " + str(self.V[j]))
                    # print("v2: " + str(v2))
                else:
                    print("sub-channel mismatch!")
                    raise
                    
                


            else:
            
                Y1 = self.V[j] + Cj
                Y2 = Fj + Ej*self.V[j]
                Y3 = Dj + self.P[j]
                Y4 = Phij - Ej*self.P[j]
    
                # print(Y1*Gj + Y2)
                self.S[j+1] = (Gj*Y3 - Y4)/(Y1*Gj + Y2)
                self.T[j+1] = (Cj*Gj - Fj)/(Y1*Gj + Y2)
                self.P[j+1] = Y3 - Y1*self.S[j+1]
                self.V[j+1] = Cj - Y1*self.T[j+1]
                    
        else:
            pass

    def forward_sweep(self, up_boundary):
        """
        前向递推（追）：计算追赶系数
        """
        # 1. 上游水位边界初始化 
        if self.BC_up == "Z_up":
            self.P[0] = up_boundary
            self.V[0] = 0.0
        elif self.BC_up == "Q_up":
            self.P[0] = up_boundary
            self.V[0] = 0.0
        else:
            pass

        C_ary, F_ary, E_ary, G_ary, D_ary, Phi_ary = self.get_discrete_coeffs(0)

        # cp(C_ary)
        
        # 2. 逐断面计算追赶系数 
        for j in range(self.N - 1):
            self.get_catchup_coefficient(C_ary[j], F_ary[j], E_ary[j], G_ary[j], D_ary[j], Phi_ary[j], self.BC_up, j)
            

    def backward_sweep(self, down_boundary):
        """
        后向回代（赶）：求解 Z 和 Q
        """
        # 1. 下游边界闭合 
        # 假设下游水位已知 Z_L2
        # plt.figure()
        # plt.plot(self.Q[0:100])
        # # plt.plot(self.T[0:100])
        # plt.show()
        if self.BC_down == "Z_down" and self.BC_up == "Q_up":
            self.Z_next[-1] = down_boundary
            self.Q_next[-1] = (self.P[-1] - self.Z_next[-1]*self.V[-1])

                # 2. 反向逐点回推
            for j in range(self.N - 2, -1, -1):
                # Qj = Sj+1 - Tj+1 * Qj+1
                self.Z_next[j] = self.S[j+1] - self.T[j+1] * self.Z_next[j+1]
                # Zj = Pj - Vj * Qj
                self.Q_next[j] = self.P[j] - self.V[j] * self.Z_next[j]
        # elif self.BC_down == "Q_down":
        #     self.Q_next[-1] = down_boundary
        #     if self.V[-1] != 0:
        #         self.Z_next[-1] = (self.P[-1] - self.Q_next[-1]) / self.V[-1]
        #     else:
        #         self.Q_next[-1] = self.Q[-1] # 兜底逻辑
        #         # 2. 反向逐点回推
        #     for j in range(self.N - 2, -1, -1):
        #         # Qj = Sj+1 - Tj+1 * Qj+1
        #         self.Z_next[j] = self.P[j+1] - self.V[j+1] * self.Z_next[j+1]
        #         # Zj = Pj - Vj * Qj
        #         self.Q_next[j] = self.P[j] - self.V[j] * self.Z_next[j]
        else: 
            pass
            
        
        

        
 
    def solve_one_step(self, up, down, iterations=3):
        """
        求解一个时间步，包含迭代修正
        """

        for _ in range(iterations):

            Z_rec = self.Z_next.copy()
            Q_rec = self.Q_next.copy()
            
            self.set_coefficients(self.Z_next, self.Q_next, self.q)
            self.forward_sweep(up)
            self.backward_sweep(down)
            self.initialize_coe()
            

            
            
            # 更新 n+1 时刻值用于下一轮迭代修正系数
            # 在实际工程中，此处会重新计算面积、摩阻等 

            Z_err = np.sum(self.Z_next - Z_rec)**2/len(Z_rec)
            Q_err = np.sum(self.Q_next - Q_rec)**2/len(Q_rec)

            # print("=====")
            # print("Z MSE: {0}".format(Z_err))
            # print("Q MSE: {0}".format(Q_err))
            # print("=====")

        # 迭代完成后，推进时间层
        self.Z = self.Z_next.copy()
        self.Q = self.Q_next.copy()
        self.update_Qf()
            
        return self.Z, self.Q, self.Qf

# 使用示例


def sim_run(Z0, Q0, Q_up, Z_down, x, t, sub_channel, par_sc):
    
    g = 9.8
    alpha = 1.0
    theta = 0.65
    n = 0.015
    
    
    n_section = len(x)
    n_time = len(t)

    # channel_length = 10000
    # x = np.array(np.linspace(0, channel_length, n_section))
    # t = np.array(np.linspace(0, 60*500, n_time))
    dx = x[1]-x[0]
    dt = t[1]-t[0]

    # [pos, [sigma_s, b, e, Z-d/Z-after gate]]
    # sub_channel = [[2000.5, [0.6, 1, e[0], 0.5]], [6000.5, [0.6, 1.5, e[1], 0.7]], [8000.5, [0.6, 1.2, e[2], 0.6]]] 
    #sub_channel = []
    
    # if len(sub_channel) != 0:
    #     xn, par_sc = make_x_array(x, sub_channel)
    #     xnn = np.array(xn)
    #     #print(par_sc)
    #     par_sc = np.array(par_sc)
    #     n_section = len(xn)
    
    
    solver = SaintVenantSolver(theta=theta, g=g, alpha=alpha, n=n, n_sections=n_section, x=x, t=t, dx=dx, dt=dt)

    if len(sub_channel) != 0:
        solver.set_sub_channel(par_sc)

    # Z0 = 0.9
    # Q0 = 0.1

    # Q_up = 2.0
    # Z_down = 0.9
    BC_up = "Q_up"
    BC_down = "Z_down"

    solver.set_boundary_condition(BC_up, BC_down)

    
    
    solver.set_initial_conditions(Z0, Q0)
    
    # 求解下一个时刻 (上游水位涨到 11m, 下游维持 10m)

    Z_evl = []
    Q_evl = []
    Qf = []
    z_max_over_time = np.full(len(x), -np.inf, dtype=np.float32)
    q_max_over_time = np.full(len(x), -np.inf, dtype=np.float32)
    qf_max_over_time = None

    for ii, tt in enumerate(t):
        up_boundary = Q_up
        down_boundary = Z_down
        Z_res, Q_res, Qf_res = solver.solve_one_step(up_boundary, down_boundary, 3)
        z_max_over_time = np.maximum(z_max_over_time, np.asarray(Z_res, dtype=np.float32))
        q_max_over_time = np.maximum(q_max_over_time, np.asarray(Q_res, dtype=np.float32))
        qf_array = np.asarray(Qf_res, dtype=np.float32)
        if qf_max_over_time is None:
            qf_max_over_time = qf_array.copy()
        else:
            qf_max_over_time = np.maximum(qf_max_over_time, qf_array)
        # print("==================================")
        # print(Qf_res)
        Qf.append(Qf_res)
        
        if ii % 120 == 0:
            Z_evl.append(Z_res)
            Q_evl.append(Q_res)

    if qf_max_over_time is None:
        qf_max_over_time = np.zeros(0, dtype=np.float32)

    safety_trace = {
        "Z_max_over_time": z_max_over_time,
        "Q_max_over_time": q_max_over_time,
        "Qf_max_over_time": qf_max_over_time,
    }

    return Z_evl, Q_evl, Qf, Z_res, Q_res, safety_trace

    # plt.figure()
    # plt.plot(Z_evl)
    # plt.show()
    
    # Z_res, Q_res = solver.solve_one_step(Q_up, Z_down, 10)
    
    # print("求解完成后的全线水位:", Z_res)


def _build_default_context():
    n_section = 600
    n_time = 432
    channel_length = 10000

    x = np.array(np.linspace(0, channel_length, n_section))
    t = np.array(np.linspace(0, 60 * 180, n_time))
    gate_specs = [
        (2000.5, [0.6, 1.0, 0.0, 0.5]),
        (6000.5, [0.6, 1.5, 0.0, 0.7]),
        (8000.5, [0.6, 1.2, 0.0, 0.6]),
    ]
    return {
        "x": x,
        "t": t,
        "Z00": 0.9,
        "Q00": 0.1,
        "Q_up": 2.0,
        "Z_down": 0.9,
        "gate_specs": gate_specs,
    }


def _build_initial_profile(value, length):
    return np.full(length, value, dtype=np.float32)


def hydraulic_simulator(e, previous_state=None, use_z00=False):
    """
    Run one hydraulic simulation period.

    Parameters
    ----------
    e : array-like
        Gate openings for each canal gate.
    previous_state : dict | None
        Previous terminal state with keys "Z" and "Q". Used when not on the first step.
    use_z00 : bool
        When True, use the default Z00/Q00 initial conditions for the first step.

    Returns
    -------
    tuple[np.ndarray, dict]
        Water supply volumes for each gate over one period, and the terminal
        hydraulic state to be reused as the next period initial condition.
    """
    context = _build_default_context()
    gate_openings = np.asarray(e, dtype=np.float32)
    gate_specs = context["gate_specs"]

    if gate_openings.shape != (len(gate_specs),):
        raise ValueError(
            f"Expected {len(gate_specs)} gate openings, got shape {gate_openings.shape}"
        )

    x = context["x"]
    t = context["t"]
    sub_channel = []
    par_sc = np.empty((0, 4), dtype=object)
    for (position, params), opening in zip(gate_specs, gate_openings):
        sigma_s, b, _, zd = params
        sub_channel.append([position, [sigma_s, b, float(opening), zd]])

    if len(sub_channel) != 0:
        xn, par_sc = make_x_array(x, sub_channel)
        x = np.array(xn)
        par_sc = np.array(par_sc, dtype=object)

    n_section = len(x)

    if use_z00 or previous_state is None:
        Z0 = _build_initial_profile(context["Z00"], n_section)
        Q0 = _build_initial_profile(context["Q00"], n_section)
    else:
        Z0 = np.asarray(previous_state["Z"], dtype=np.float32).copy()
        Q0 = np.asarray(previous_state["Q"], dtype=np.float32).copy()
        if Z0.shape != (n_section,) or Q0.shape != (n_section,):
            raise ValueError(
                "Previous hydraulic state shape does not match the expanded x grid: "
                f"expected {(n_section,)}, got Z={Z0.shape}, Q={Q0.shape}."
            )

    Q_up = context["Q_up"]
    Z_down = context["Z_down"]

    t_st = time.time()
    Z_evl, Q_evl, Qf, Z_res, Q_res, safety_trace = sim_run(
        Z0, Q0, Q_up, Z_down, x, t, sub_channel, par_sc
    )
    t_ed = time.time()

    combined = np.array(Qf, dtype=np.float32)
    if combined.size == 0:
        water_volume = np.zeros(len(gate_openings), dtype=np.float32)
    else:
        Qf_res = combined.transpose()
        water_volume = []
        for ii in range(len(Qf_res)):
            water_volume.append(np.trapezoid(Qf_res[ii], x=t))
        water_volume = np.asarray(water_volume, dtype=np.float32)

    final_state = {
        "Z": np.asarray(Z_res, dtype=np.float32),
        "Q": np.asarray(Q_res, dtype=np.float32),
        "Qf": water_volume.copy(),
        "Z_max_over_time": np.asarray(safety_trace["Z_max_over_time"], dtype=np.float32),
        "Q_max_over_time": np.asarray(safety_trace["Q_max_over_time"], dtype=np.float32),
        "Qf_max_over_time": np.asarray(safety_trace["Qf_max_over_time"], dtype=np.float32),
        "elapsed_seconds": float(t_ed - t_st),
    }
    return water_volume, final_state
