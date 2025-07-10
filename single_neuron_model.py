import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    """高斯伪导数的spike函数"""
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude
        de_dv_scaled = de_dz * dz_dv_scaled
        return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name='spike_gauss'), grad


class SingleNeuronModel:
    """
    基于BillehColumn动力学的单神经元模型
    完全保持相同的动力学机制，但简化为单个神经元
    """
    
    def __init__(self, params, dt=1.0, gauss_std=0.5, dampening_factor=0.3):
        """
        初始化单神经元模型
        
        参数:
        params: 神经元参数字典
        dt: 时间步长 (ms) - 与BillehColumn保持一致，默认1.0
        gauss_std: 高斯伪导数标准差
        dampening_factor: 阻尼因子
        """
        self._dt = dt
        self._gauss_std = gauss_std
        self._dampening_factor = dampening_factor
        
        # 复制原始参数 (与BillehColumn完全一致的处理)
        self._params = {}
        for key, value in params.items():
            self._params[key] = value.copy()
        
        # 电压归一化 (与BillehColumn完全一致)
        voltage_scale = self._params['V_th'] - self._params['E_L']
        voltage_offset = self._params['E_L']
        self._params['V_th'] = (self._params['V_th'] - voltage_offset) / voltage_scale
        self._params['E_L'] = (self._params['E_L'] - voltage_offset) / voltage_scale
        self._params['V_reset'] = (self._params['V_reset'] - voltage_offset) / voltage_scale
        self._params['asc_amps'] = self._params['asc_amps'] / voltage_scale[..., None]
        
        # 计算衍生参数 (与BillehColumn完全一致)
        tau = self._params['C_m'] / self._params['g']
        self._decay = np.exp(-dt / tau)
        self._current_factor = 1 / self._params['C_m'] * (1 - self._decay) * tau
        self._syn_decay = np.exp(-dt / np.array(self._params['tau_syn']))
        self._psc_initial = np.e / np.array(self._params['tau_syn'])
        
        # 保存缩放参数
        self.voltage_scale = voltage_scale[0]
        self.voltage_offset = voltage_offset[0]
        
        # 转换为标量参数 (单神经元)
        self.v_th = self._params['V_th'][0]
        self.e_l = self._params['E_L'][0]
        self.v_reset = self._params['V_reset'][0]
        self.param_g = self._params['g'][0]
        self.t_ref = self._params['t_ref'][0]
        self.k = self._params['k'][0]
        self.asc_amps = self._params['asc_amps'][0]
        self.decay = self._decay[0]
        self.current_factor = self._current_factor[0]
        self.syn_decay = self._syn_decay[0]
        self.psc_initial = self._psc_initial[0]
        
        # 突触数量
        self.n_receptors = self._params['tau_syn'].shape[1]
        
        # 初始化状态
        self.reset_state()
    
    def reset_state(self):
        """重置神经元状态 (与BillehColumn的zero_state一致)"""
        self.v = self.v_th * 0.0 + 1.0 * self.v_reset  # 归一化电压
        self.r = 0.0  # 不应期计数器
        self.asc_1 = 0.0  # adaptation电流1
        self.asc_2 = 0.0  # adaptation电流2
        self.psc_rise = np.zeros(self.n_receptors)  # 突触电流上升
        self.psc = np.zeros(self.n_receptors)  # 突触电流
    
    def step(self, input_current):
        """
        执行一个时间步 (与BillehColumn的call方法完全一致)
        
        参数:
        input_current: 输入电流 (pA)
        
        返回:
        spike: 是否产生spike (0或1)
        voltage: 实际电压值 (mV)
        """
        # 保存上一步的spike状态
        prev_z = getattr(self, 'prev_z', 0)
        
        # 更新突触电流 (没有外部突触输入，只有递归连接)
        # 对于单神经元，没有递归连接，所以突触输入为0
        rec_inputs = np.zeros(self.n_receptors)
        
        # 完全按照BillehColumn的顺序更新突触电流
        new_psc_rise = self.syn_decay * self.psc_rise + rec_inputs * self.psc_initial
        new_psc = self.psc * self.syn_decay + self._dt * self.syn_decay * self.psc_rise
        
        # 更新不应期计数器 (与BillehColumn完全一致: tf.nn.relu)
        new_r = max(0, self.r + prev_z * self.t_ref - self._dt)
        
        # 更新adaptation电流 (与BillehColumn完全一致)
        new_asc_1 = np.exp(-self._dt * self.k[0]) * self.asc_1 + prev_z * self.asc_amps[0]
        new_asc_2 = np.exp(-self._dt * self.k[1]) * self.asc_2 + prev_z * self.asc_amps[1]
        
        # 计算电压 (与BillehColumn完全一致，但加上外部输入电流)
        reset_current = prev_z * (self.v_reset - self.v_th)
        input_current_sum = np.sum(self.psc)  # 注意：使用当前时刻的psc而不是new_psc
        decayed_v = self.decay * self.v
        
        gathered_g = self.param_g * self.e_l
        # 将外部输入电流(pA)转换为与模型一致的单位
        # 输入电流需要除以voltage_scale来归一化
        external_current = input_current / self.voltage_scale
        c1 = input_current_sum + self.asc_1 + self.asc_2 + gathered_g + external_current
        new_v = decayed_v + self.current_factor * c1 + reset_current
        
        # 检查是否产生spike (与BillehColumn完全一致)
        normalizer = self.v_th - self.e_l
        v_sc = (new_v - self.v_th) / normalizer
        
        # # 使用spike_gauss函数的逻辑 (与BillehColumn完全一致)
        new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)        
        new_z = tf.where(new_r > 0., tf.zeros_like(new_z), new_z)
        
        # 更新状态
        self.psc_rise = new_psc_rise
        self.psc = new_psc
        self.r = new_r
        self.asc_1 = new_asc_1
        self.asc_2 = new_asc_2
        self.v = new_v
        
        # 保存当前spike状态供下一步使用
        self.prev_z = new_z
        
        return new_z, new_v * self.voltage_scale + self.voltage_offset
    
    def simulate(self, duration, input_current):
        """
        模拟一段时间
        
        参数:
        duration: 模拟时长 (ms)
        input_current: 输入电流 (pA)，可以是常数或数组
        
        返回:
        time: 时间数组
        spikes: spike数组
        voltages: 电压数组
        """
        n_steps = int(duration / self._dt)
        time = np.arange(n_steps) * self._dt
        
        # 处理输入电流
        if np.isscalar(input_current):
            current = np.full(n_steps, input_current)
        else:
            current = input_current
            
        spikes = np.zeros(n_steps)
        voltages = np.zeros(n_steps)
        
        for i in range(n_steps):
            spike, voltage = self.step(current[i])
            spikes[i] = spike
            voltages[i] = voltage
            
        return time, spikes, voltages


def create_neuron_with_params():
    """使用文件中第516行的参数创建神经元"""
    params = {
        'asc_init': np.array([[0., 0.]]),
        'V_th': np.array([-34.78002413]),
        'g': np.array([4.33266634]),
        'E_L': np.array([-71.31963094]),
        'k': np.array([[0.003, 0.03]]),
        'C_m': np.array([61.77601314]),
        'V_reset': np.array([-71.31963094]),
        'V_dynamics_method': np.array(['linear_exact'], dtype='<U12'),
        'tau_syn': np.array([[5.5, 8.5, 2.8, 5.8]]),
        't_ref': np.array([2.2]),
        'asc_amps': np.array([[-6.62149399, -68.56339311]])
    }
    
    return SingleNeuronModel(params, dt=1.0)


def test_neuron_with_single_current(current_pA):
    """测试单个电流值"""
    # 创建神经元
    neuron = create_neuron_with_params()
    
    # 模拟参数
    duration = 2000  # ms
    input_current = current_pA  # pA
    
    # 运行模拟
    time, spikes, voltages = neuron.simulate(duration, input_current)
    
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 绘制电压轨迹
    ax1.plot(time, voltages, 'b-', linewidth=0.8)
    ax1.set_ylabel('电压 (mV)')
    ax1.set_title(f'单神经元模型 - {current_pA}pA 输入电流')
    ax1.grid(True, alpha=0.3)
    
    # 绘制spike
    spike_times = time[spikes > 0]
    spike_heights = voltages[spikes > 0]
    ax2.plot(spike_times, np.ones_like(spike_times), 'r|', markersize=10, markeredgewidth=2)
    ax2.set_ylabel('Spike')
    ax2.set_xlabel('时间 (ms)')
    ax2.set_ylim(0, 1.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'single_neuron_firing_{current_pA}pA.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 计算firing rate
    firing_rate = len(spike_times) / (duration / 1000)  # Hz
    print(f"发放率: {firing_rate:.2f} Hz")
    print(f"总共产生了 {len(spike_times)} 个action potential")
    
    return time, spikes, voltages


def compute_firing_rate(neuron, current_pA, duration=3000):
    """计算给定电流下的发放率"""
    # 运行模拟
    time, spikes, voltages = neuron.simulate(duration, current_pA)
    
    # 计算firing rate，忽略前500ms的适应期
    settle_time = 500  # ms
    settle_steps = int(settle_time / neuron._dt)
    
    if len(spikes) > settle_steps:
        valid_spikes = spikes[settle_steps:]
        valid_duration = (len(valid_spikes) * neuron._dt) / 1000  # 转换为秒
        firing_rate = np.sum(valid_spikes) / valid_duration
    else:
        firing_rate = 0
    
    return firing_rate


def plot_FI_curve(fast_mode=False):
    """绘制F-I曲线（频率-电流关系）"""
    print("正在计算F-I曲线...")
    
    # 创建神经元
    neuron = create_neuron_with_params()
    
    # 打印神经元参数
    print(f"\n神经元参数:")
    print(f"V_th = {neuron.v_th * neuron.voltage_scale + neuron.voltage_offset:.2f} mV")
    print(f"V_reset = {neuron.v_reset * neuron.voltage_scale + neuron.voltage_offset:.2f} mV")
    print(f"E_L = {neuron.e_l * neuron.voltage_scale + neuron.voltage_offset:.2f} mV")
    print(f"t_ref = {neuron.t_ref:.2f} ms")
    print(f"C_m = {neuron._params['C_m'][0]:.2f} pF")
    print(f"g = {neuron._params['g'][0]:.2f} nS")
    
    # 电流范围：0到250pA
    if fast_mode:
        currents = np.arange(0, 251, 25)  # 每25pA一个点，快速模式
        print("使用快速模式 (每25pA一个点)")
    else:
        currents = np.arange(0, 251, 10)  # 每10pA一个点，标准模式
        print("使用标准模式 (每10pA一个点)")
    
    firing_rates = []
    
    for i, current in enumerate(currents):
        print(f"正在计算电流 {current} pA ({i+1}/{len(currents)})")
        
        # 重置神经元状态
        neuron.reset_state()
        
        # 计算发放率
        firing_rate = compute_firing_rate(neuron, current)
        firing_rates.append(firing_rate)
        
        print(f"  发放率: {firing_rate:.2f} Hz")
    
    # 绘制F-I曲线
    plt.figure(figsize=(10, 6))
    plt.plot(currents, firing_rates, 'bo-', linewidth=2, markersize=4)
    plt.xlabel('输入电流 (pA)')
    plt.ylabel('发放率 (Hz)')
    plt.title('F-I曲线：发放率与输入电流的关系')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 250)
    plt.ylim(0, max(firing_rates) * 1.1 if max(firing_rates) > 0 else 10)
    
    # 添加一些统计信息
    plt.text(0.02, 0.98, f'神经元参数:\nV_th = {neuron.v_th * neuron.voltage_scale + neuron.voltage_offset:.1f} mV\nV_reset = {neuron.v_reset * neuron.voltage_scale + neuron.voltage_offset:.1f} mV\nt_ref = {neuron.t_ref:.1f} ms', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('FI_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 保存数据
    np.savetxt('FI_curve_data.txt', np.column_stack([currents, firing_rates]), 
               header='Current(pA) FiringRate(Hz)', fmt='%.2f')
    
    print(f"\nF-I曲线数据已保存到 FI_curve_data.txt")
    print(f"最大发放率: {max(firing_rates):.2f} Hz (在 {currents[np.argmax(firing_rates)]} pA)")
    
    return currents, firing_rates


if __name__ == "__main__":
    # 绘制F-I曲线
    plot_FI_curve()
    
    # 如果需要测试单个电流值，可以调用以下函数
    test_neuron_with_single_current(250)  # 测试100pA 