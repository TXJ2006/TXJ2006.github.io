(() => {
  const page = document.querySelector('.editor-page');
  if (!page) return;

  const repo = page.dataset.repo;
  const privateRepo = page.dataset.privateRepo;
  const owner = page.dataset.owner;
  const branch = page.dataset.branch;
  const siteUrl = page.dataset.siteUrl.replace(/\/$/, '');
  const apiRoot = `https://api.github.com/repos/${repo}`;
  const privateApiRoot = `https://api.github.com/repos/${privateRepo}`;
  const tokenKey = 'txj_notes_editor_token';
  const draftsKey = 'txj_notes_editor_drafts';
  const params = new URLSearchParams(window.location.search);

  const mathMacros = {
    '\\E': '\\mathbb{E}',
    '\\Pbb': '\\mathbb{P}',
    '\\Pp': '\\mathbb{P}',
    '\\Prob': '\\mathbb{P}',
    '\\KL': '\\operatorname{KL}',
    '\\kl': '\\operatorname{kl}',
    '\\Ber': '\\operatorname{Bern}',
    '\\Beta': '\\operatorname{Beta}',
    '\\Poi': '\\operatorname{Poisson}',
    '\\Normal': '\\mathcal{N}',
    '\\Var': '\\operatorname{Var}',
    '\\Cov': '\\operatorname{Cov}',
    '\\Reg': '\\operatorname{Reg}',
    '\\TV': '\\operatorname{TV}',
    '\\Alt': '\\operatorname{Alt}',
    '\\argmax': '\\operatorname*{arg\\,max}',
    '\\argmin': '\\operatorname*{arg\\,min}',
    '\\dd': '\\mathrm{d}',
    '\\one': '\\mathbf{1}',
    '\\ind': '\\mathbf{1}',
    '\\R': '\\mathbb{R}',
    '\\F': '\\mathcal{F}',
    '\\G': '\\mathcal{G}',
    '\\GP': '\\mathcal{GP}',
    '\\given': '\\,\\middle|\\,',
    '\\st': '\\text{subject to}',
    '\\bm': '\\boldsymbol{#1}',
    '\\label': '\\phantom{#1}',
    '\\midrule': '\\hline',
    '\\calA': '\\mathcal{A}',
    '\\calD': '\\mathcal{D}',
    '\\calF': '\\mathcal{F}',
    '\\calH': '\\mathcal{H}',
    '\\calX': '\\mathcal{X}',
  };

  const formulaCatalog = {
    '常用结构': [
      ['a/b', '普通分式', '\\frac{a}{b}'], ['x²', '上标', '{{selection}}^{2}'],
      ['xᵢ', '下标', '{{selection}}_{i}'], ['xᵢ²', '上下标', '{{selection}}_{i}^{2}'],
      ['√x', '平方根', '\\sqrt{{{selection}}}'], ['ⁿ√x', 'n 次根', '\\sqrt[n]{{{selection}}}'],
      ['|x|', '绝对值', '\\left|{{selection}}\\right|'], ['‖x‖', '范数', '\\left\\|{{selection}}\\right\\|'],
      ['⟨x,y⟩', '内积', '\\langle x,y\\rangle'], ['(n k)', '二项式系数', '\\binom{n}{k}'],
      ['Σ', '求和', '\\sum_{i=1}^{n} {{selection}}'], ['Π', '连乘', '\\prod_{i=1}^{n} {{selection}}'],
      ['∫', '定积分', '\\int_a^b {{selection}}\\,\\dd x'], ['lim', '极限', '\\lim_{n\\to\\infty} {{selection}}'],
      ['d/dx', '导数', '\\frac{\\dd}{\\dd x}{{selection}}'], ['∂/∂x', '偏导数', '\\frac{\\partial}{\\partial x}{{selection}}'],
      ['cases', '分段函数', 'f(x)=\\begin{cases}x^2,&x\\ge 0,\\\\-x,&x<0.\\end{cases}'],
      ['align', '对齐公式', '\\begin{aligned}a&=b+c,\\\\d&=e+f.\\end{aligned}'],
      ['2×2', '二阶矩阵', '\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}'],
      ['3×3', '三阶矩阵', '\\begin{pmatrix}a&b&c\\\\d&e&f\\\\g&h&i\\end{pmatrix}'],
      ['→', '趋于', '\\to'], ['⇒', '推出', '\\Rightarrow'], ['∈', '属于', '\\in'], ['∀', '任意', '\\forall'],
    ],
    '分式': [
      ['a/b', '普通分式', '\\frac{a}{b}'], ['a⁄b', '显示型分式', '\\dfrac{a}{b}'],
      ['a/bₛ', '文本型分式', '\\tfrac{a}{b}'], ['1/x', '倒数', '\\frac{1}{{{selection}}}'],
      ['dx/dt', '微分商', '\\frac{\\dd x}{\\dd t}'], ['∂f/∂x', '偏微分商', '\\frac{\\partial f}{\\partial x}'],
      ['Δy/Δx', '差商', '\\frac{\\Delta y}{\\Delta x}'], ['a/(b/c)', '嵌套分式', '\\frac{a}{\\frac{b}{c}}'],
      ['(a/b)/(c/d)', '复合分式', '\\frac{\\frac{a}{b}}{\\frac{c}{d}}'],
      ['Σ/Π', '求和除以连乘', '\\frac{\\sum_{i=1}^{n}a_i}{\\prod_{j=1}^{m}b_j}'],
      ['binom', '二项式系数', '\\binom{n}{k}'], ['1/(1+)', '连分式', 'a_0+\\cfrac{1}{a_1+\\cfrac{1}{a_2}}'],
      ['dy/dx|₀', '指定点导数', '\\left.\\frac{\\dd y}{\\dd x}\\right|_{x=x_0}'],
      ['P(A|B)', '条件概率分式', '\\frac{\\Pbb(A\\cap B)}{\\Pbb(B)}'],
    ],
    '上下标': [
      ['x²', '上标', '{{selection}}^{2}'], ['xⁿ', 'n 次幂', '{{selection}}^{n}'],
      ['x⁻¹', '逆或负一次幂', '{{selection}}^{-1}'], ['xᵢ', '下标', '{{selection}}_{i}'],
      ['xᵢⱼ', '多重下标', '{{selection}}_{ij}'], ['xᵢ²', '上下标', '{{selection}}_{i}^{2}'],
      ['xᵢ₌₁ⁿ', '上下限', '{{selection}}_{i=1}^{n}'], ['ₐXᵇ', '左上下标', '{}_{a}^{b}{{selection}}'],
      ['x₍ᵢⱼ₎', '括号下标', '{{selection}}_{(i,j)}'], ['xₜ₋₁', '时间滞后下标', '{{selection}}_{t-1}'],
      ['x⁽ⁿ⁾', 'n 阶导数上标', '{{selection}}^{(n)}'], ['eˣ', '指数函数', 'e^{{{selection}}}'],
      ['aₙₖ', '双指标', 'a_{n,k}'], ['Σᵢ₌₁ⁿ', '求和上下限', '\\sum_{i=1}^{n}{{selection}}'],
    ],
    '根式': [
      ['√x', '平方根', '\\sqrt{{{selection}}}'], ['∛x', '立方根', '\\sqrt[3]{{{selection}}}'],
      ['ⁿ√x', 'n 次根', '\\sqrt[n]{{{selection}}}'], ['√a/b', '根式分式', '\\sqrt{\\frac{a}{b}}'],
      ['1/√x', '根式倒数', '\\frac{1}{\\sqrt{{{selection}}}}'], ['√(a+b)', '和的平方根', '\\sqrt{a+b}'],
      ['√Σ', '求和的平方根', '\\sqrt{\\sum_{i=1}^{n}x_i^2}'], ['√√x', '嵌套根式', '\\sqrt{a+\\sqrt{b}}'],
      ['√[m]{xⁿ}', '幂的 m 次根', '\\sqrt[m]{x^n}'], ['±√x', '正负平方根', '\\pm\\sqrt{{{selection}}}'],
      ['‖x‖₂', '二范数根式', '\\left(\\sum_{i=1}^{n}x_i^2\\right)^{1/2}'],
      ['geom', '几何平均', '\\sqrt[n]{\\prod_{i=1}^{n}x_i}'],
    ],
    '积分与微分': [
      ['∫f dx', '不定积分', '\\int {{selection}}\\,\\dd x'], ['∫ₐᵇ', '定积分', '\\int_a^b {{selection}}\\,\\dd x'],
      ['∫₀∞', '无穷区间积分', '\\int_0^{\\infty} {{selection}}\\,\\dd x'],
      ['∬D', '二重积分', '\\iint_D {{selection}}\\,\\dd x\\,\\dd y'],
      ['∭V', '三重积分', '\\iiint_V {{selection}}\\,\\dd x\\,\\dd y\\,\\dd z'],
      ['∮γ', '曲线积分', '\\oint_{\\gamma} {{selection}}\\,\\dd z'],
      ['∮F·dr', '向量线积分', '\\oint_C \\mathbf{F}\\cdot\\dd\\mathbf{r}'],
      ['∫S', '曲面积分', '\\iint_S {{selection}}\\,\\dd S'],
      ['d/dx', '一阶导数算子', '\\frac{\\dd}{\\dd x}{{selection}}'], ['d²/dx²', '二阶导数', '\\frac{\\dd^2}{\\dd x^2}{{selection}}'],
      ['f′', '一阶撇导数', "{{selection}}'"], ['f″', '二阶撇导数', "{{selection}}''"],
      ['∂/∂x', '一阶偏导', '\\frac{\\partial}{\\partial x}{{selection}}'],
      ['∂²/∂x²', '二阶偏导', '\\frac{\\partial^2}{\\partial x^2}{{selection}}'],
      ['∂²/∂x∂y', '混合偏导', '\\frac{\\partial^2 {{selection}}}{\\partial x\\,\\partial y}'],
      ['∇f', '梯度', '\\nabla {{selection}}'], ['∇·F', '散度', '\\nabla\\cdot\\mathbf{F}'],
      ['∇×F', '旋度', '\\nabla\\times\\mathbf{F}'], ['Δf', '拉普拉斯算子', '\\Delta {{selection}}'],
      ['Jf', '雅可比矩阵', 'J_f(x)=\\left[\\frac{\\partial f_i}{\\partial x_j}\\right]_{ij}'],
      ['Hf', '海森矩阵', 'H_f(x)=\\left[\\frac{\\partial^2 f}{\\partial x_i\\partial x_j}\\right]_{ij}'],
      ['δF/δu', '变分导数', '\\frac{\\delta F}{\\delta u}'], ['Dᵥf', '方向导数', 'D_v f(x)=\\nabla f(x)\\cdot v'],
    ],
    '大型运算符': [
      ['Σ', '求和', '\\sum_{i=1}^{n} {{selection}}'], ['Σ∞', '无穷求和', '\\sum_{n=0}^{\\infty} {{selection}}'],
      ['Π', '连乘', '\\prod_{i=1}^{n} {{selection}}'], ['∐', '余积', '\\coprod_{i\\in I} X_i'],
      ['⋃', '大并集', '\\bigcup_{i\\in I} A_i'], ['⋂', '大交集', '\\bigcap_{i\\in I} A_i'],
      ['⨁', '大直和', '\\bigoplus_{i\\in I} V_i'], ['⨂', '大张量积', '\\bigotimes_{i=1}^{n} V_i'],
      ['⋁', '大析取', '\\bigvee_{i\\in I} P_i'], ['⋀', '大合取', '\\bigwedge_{i\\in I} P_i'],
      ['lim', '极限', '\\lim_{x\\to a} {{selection}}'], ['lim∞', '无穷远极限', '\\lim_{n\\to\\infty} {{selection}}'],
      ['lim sup', '上极限', '\\limsup_{n\\to\\infty} {{selection}}'], ['lim inf', '下极限', '\\liminf_{n\\to\\infty} {{selection}}'],
      ['max', '最大值', '\\max_{x\\in X} {{selection}}'], ['min', '最小值', '\\min_{x\\in X} {{selection}}'],
      ['sup', '上确界', '\\sup_{x\\in X} {{selection}}'], ['inf', '下确界', '\\inf_{x\\in X} {{selection}}'],
      ['arg max', '最大值点', '\\argmax_{x\\in X} {{selection}}'], ['arg min', '最小值点', '\\argmin_{x\\in X} {{selection}}'],
      ['ess sup', '本质上确界', '\\operatorname*{ess\\,sup}_{x\\in X} {{selection}}'],
    ],
    '括号与定界符': [
      ['(x)', '圆括号', '\\left({{selection}}\\right)'], ['[x]', '方括号', '\\left[{{selection}}\\right]'],
      ['{x}', '花括号', '\\left\\{{{selection}}\\right\\}'], ['⟨x⟩', '尖括号', '\\langle{{selection}}\\rangle'],
      ['|x|', '绝对值', '\\left|{{selection}}\\right|'], ['‖x‖', '双竖线范数', '\\left\\|{{selection}}\\right\\|'],
      ['⌊x⌋', '向下取整', '\\lfloor{{selection}}\\rfloor'], ['⌈x⌉', '向上取整', '\\lceil{{selection}}\\rceil'],
      ['[a,b]', '闭区间', '[a,b]'], ['(a,b)', '开区间', '(a,b)'], ['[a,b)', '左闭右开区间', '[a,b)'],
      ['{x:P}', '集合描述', '\\left\\{x\\in X\\middle|P(x)\\right\\}'],
      ['⟨x|y⟩', '狄拉克内积', '\\left\\langle x\\middle|y\\right\\rangle'],
      ['f|ₐᵇ', '上下限代入', '\\left.{{selection}}\\right|_{a}^{b}'],
      ['(a/b)', '自动缩放分式括号', '\\left(\\frac{a}{b}\\right)'],
      ['‖A‖F', 'Frobenius 范数', '\\left\\|A\\right\\|_{\\mathrm F}'],
    ],
    '函数': [
      ['sin', '正弦', '\\sin({{selection}})'], ['cos', '余弦', '\\cos({{selection}})'],
      ['tan', '正切', '\\tan({{selection}})'], ['cot', '余切', '\\cot({{selection}})'],
      ['sec', '正割', '\\sec({{selection}})'], ['csc', '余割', '\\csc({{selection}})'],
      ['arcsin', '反正弦', '\\arcsin({{selection}})'], ['arccos', '反余弦', '\\arccos({{selection}})'],
      ['arctan', '反正切', '\\arctan({{selection}})'], ['sinh', '双曲正弦', '\\sinh({{selection}})'],
      ['cosh', '双曲余弦', '\\cosh({{selection}})'], ['tanh', '双曲正切', '\\tanh({{selection}})'],
      ['log', '对数', '\\log({{selection}})'], ['logₐ', '以 a 为底的对数', '\\log_a({{selection}})'],
      ['ln', '自然对数', '\\ln({{selection}})'], ['exp', '指数函数', '\\exp({{selection}})'],
      ['gcd', '最大公因数', '\\gcd(a,b)'], ['lcm', '最小公倍数', '\\operatorname{lcm}(a,b)'],
      ['sgn', '符号函数', '\\operatorname{sgn}({{selection}})'], ['Re', '实部', '\\operatorname{Re}(z)'],
      ['Im', '虚部', '\\operatorname{Im}(z)'], ['arg', '辐角', '\\arg(z)'],
      ['mod', '模运算', 'a\\bmod n'], ['pmod', '同余模数', 'a\\equiv b\\pmod n'],
    ],
    '重音与标记': [
      ['x̂', '帽子', '\\hat{{{selection}}}'], ['x̂wide', '宽帽子', '\\widehat{{{selection}}}'],
      ['x̃', '波浪号', '\\tilde{{{selection}}}'], ['x̃wide', '宽波浪号', '\\widetilde{{{selection}}}'],
      ['x̄', '短上划线', '\\bar{{{selection}}}'], ['x̅', '长上划线', '\\overline{{{selection}}}'],
      ['x̲', '下划线', '\\underline{{{selection}}}'], ['x⃗', '向量箭头', '\\vec{{{selection}}}'],
      ['AB→', '长右向量', '\\overrightarrow{{{selection}}}'], ['AB←', '长左向量', '\\overleftarrow{{{selection}}}'],
      ['ẋ', '一点', '\\dot{{{selection}}}'], ['ẍ', '两点', '\\ddot{{{selection}}}'],
      ['x̆', '短音符', '\\breve{{{selection}}}'], ['x̌', '倒帽子', '\\check{{{selection}}}'],
      ['x́', '锐音符', '\\acute{{{selection}}}'], ['x̀', '钝音符', '\\grave{{{selection}}}'],
      ['x̊', '圆环', '\\mathring{{{selection}}}'], ['overbrace', '上大括号', '\\overbrace{{{selection}}}^{\\text{说明}}'],
      ['underbrace', '下大括号', '\\underbrace{{{selection}}}_{\\text{说明}}'],
      ['boxed', '公式方框', '\\boxed{{{selection}}}'], ['cancel', '删除线', '\\cancel{{{selection}}}'],
      ['overset', '上方标注', '\\overset{*}{{{selection}}}'], ['underset', '下方标注', '\\underset{n\\to\\infty}{{{selection}}}'],
    ],
    '希腊字母': [
      ['α', 'alpha', '\\alpha'], ['β', 'beta', '\\beta'], ['γ', 'gamma', '\\gamma'],
      ['δ', 'delta', '\\delta'], ['ε', 'epsilon', '\\epsilon'], ['ϵ', 'varepsilon', '\\varepsilon'], ['ζ', 'zeta', '\\zeta'],
      ['η', 'eta', '\\eta'], ['θ', 'theta', '\\theta'], ['ϑ', 'vartheta', '\\vartheta'], ['ι', 'iota', '\\iota'], ['κ', 'kappa', '\\kappa'],
      ['λ', 'lambda', '\\lambda'], ['μ', 'mu', '\\mu'], ['ν', 'nu', '\\nu'],
      ['ξ', 'xi', '\\xi'], ['π', 'pi', '\\pi'], ['ϖ', 'varpi', '\\varpi'], ['ρ', 'rho', '\\rho'], ['ϱ', 'varrho', '\\varrho'],
      ['σ', 'sigma', '\\sigma'], ['ς', 'varsigma', '\\varsigma'], ['τ', 'tau', '\\tau'], ['υ', 'upsilon', '\\upsilon'], ['φ', 'phi', '\\phi'], ['ϕ', 'varphi', '\\varphi'],
      ['χ', 'chi', '\\chi'], ['ψ', 'psi', '\\psi'], ['ω', 'omega', '\\omega'],
      ['Γ', 'Gamma', '\\Gamma'], ['Δ', 'Delta', '\\Delta'], ['Θ', 'Theta', '\\Theta'],
      ['Λ', 'Lambda', '\\Lambda'], ['Ξ', 'Xi', '\\Xi'], ['Π', 'Pi', '\\Pi'], ['Σ', 'Sigma', '\\Sigma'], ['Υ', 'Upsilon', '\\Upsilon'], ['Φ', 'Phi', '\\Phi'],
      ['Ψ', 'Psi', '\\Psi'], ['Ω', 'Omega', '\\Omega'],
    ],
    '基础运算': [
      ['Σ', 'Summation', '\\sum_{i=1}^{n} {{selection}}'],
      ['Π', 'Product', '\\prod_{i=1}^{n} {{selection}}'],
      ['∫', 'Integral', '\\int_a^b {{selection}}\\,\\dd x'],
      ['∬', 'Double integral', '\\iint_D {{selection}}\\,\\dd x\\,\\dd y'],
      ['∮', 'Contour integral', '\\oint_\\gamma {{selection}}\\,\\dd z'],
      ['lim', 'Limit', '\\lim_{n\\to\\infty} {{selection}}'],
      ['d/dx', 'Derivative', '\\frac{\\dd}{\\dd x}{{selection}}'],
      ['∂/∂x', 'Partial derivative', '\\frac{\\partial}{\\partial x}{{selection}}'],
      ['∇', 'Gradient', '\\nabla {{selection}}'],
      ['Δ', 'Laplacian', '\\Delta {{selection}}'],
      ['∞', 'Infinity', '\\infty'], ['max', 'Maximum', '\\max_{x\\in X} {{selection}}'],
      ['min', 'Minimum', '\\min_{x\\in X} {{selection}}'],
      ['arg max', 'Arg maximum', '\\argmax_{x\\in X} {{selection}}'],
      ['arg min', 'Arg minimum', '\\argmin_{x\\in X} {{selection}}'],
    ],
    '关系符号': [
      ['=', 'Equal', '='], ['≠', 'Not equal', '\\ne'], ['≈', 'Approximately', '\\approx'],
      ['∼', 'Similar', '\\sim'], ['≅', 'Congruent', '\\cong'], ['≤', 'Less than or equal', '\\le'],
      ['≥', 'Greater than or equal', '\\ge'], ['≪', 'Much less than', '\\ll'],
      ['≫', 'Much greater than', '\\gg'], ['∝', 'Proportional', '\\propto'],
      ['⊥', 'Perpendicular', '\\perp'], ['∥', 'Parallel', '\\parallel'],
      ['≡', 'Equivalent', '\\equiv'], [':=', 'Defined as', ':='],
    ],
    '集合与逻辑': [
      ['∈', 'Element of', '\\in'], ['∉', 'Not an element of', '\\notin'],
      ['⊂', 'Subset', '\\subset'], ['⊆', 'Subset or equal', '\\subseteq'],
      ['⊃', 'Superset', '\\supset'], ['⊇', 'Superset or equal', '\\supseteq'],
      ['∪', 'Union', '\\cup'], ['∩', 'Intersection', '\\cap'],
      ['∅', 'Empty set', '\\varnothing'], ['∖', 'Set difference', '\\setminus'],
      ['∀', 'For all', '\\forall'], ['∃', 'There exists', '\\exists'],
      ['ℕ', 'Natural numbers', '\\mathbb{N}'], ['ℤ', 'Integers', '\\mathbb{Z}'],
      ['ℚ', 'Rationals', '\\mathbb{Q}'], ['ℝ', 'Real numbers', '\\mathbb{R}'],
      ['ℂ', 'Complex numbers', '\\mathbb{C}'],
    ],
    '箭头': [
      ['→', 'Right arrow', '\\to'], ['←', 'Left arrow', '\\leftarrow'],
      ['↔', 'Left-right arrow', '\\leftrightarrow'], ['⇒', 'Implies', '\\Rightarrow'],
      ['⇐', 'Implied by', '\\Leftarrow'], ['⇔', 'If and only if', '\\Longleftrightarrow'],
      ['↦', 'Maps to', '\\mapsto'], ['↑', 'Up arrow', '\\uparrow'],
      ['↓', 'Down arrow', '\\downarrow'], ['⟶', 'Long right arrow', '\\longrightarrow'],
      ['⇀', 'Weak convergence', '\\rightharpoonup'], ['↗', 'North-east arrow', '\\nearrow'],
    ],
    '矩阵与线性代数': [
      ['v⃗', 'Vector', '\\vec{{{selection}}}'], ['𝐱', 'Bold vector', '\\bm{{{selection}}}'],
      ['Aᵀ', 'Transpose', 'A^{\\mathsf T}'], ['A⁻¹', 'Inverse', 'A^{-1}'],
      ['⟨x,y⟩', 'Inner product', '\\langle x,y\\rangle'],
      ['2×2', '2 by 2 matrix', '\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}'],
      ['3×3', '3 by 3 matrix', '\\begin{pmatrix}a&b&c\\\\d&e&f\\\\g&h&i\\end{pmatrix}'],
      ['det', 'Determinant', '\\det(A)'], ['tr', 'Trace', '\\operatorname{tr}(A)'],
      ['rank', 'Rank', '\\operatorname{rank}(A)'], ['ker', 'Kernel', '\\ker(A)'],
      ['span', 'Span', '\\operatorname{span}\\{v_1,\\ldots,v_n\\}'],
      ['⊗', 'Tensor product', '\\otimes'], ['⊕', 'Direct sum', '\\oplus'],
    ],
    '数学字体': [
      ['𝐱', '粗体', '\\mathbf{{{selection}}}'], ['𝒙', '粗斜体向量', '\\boldsymbol{{{selection}}}'],
      ['𝑥', '数学斜体', '\\mathit{{{selection}}}'], ['x', '正体', '\\mathrm{{{selection}}}'],
      ['𝖷', '无衬线体', '\\mathsf{{{selection}}}'], ['𝚡', '打字机体', '\\mathtt{{{selection}}}'],
      ['𝒜', '花体', '\\mathcal{{{selection}}}'], ['𝔄', '哥特体', '\\mathfrak{{{selection}}}'],
      ['𝔸', '黑板粗体', '\\mathbb{{{selection}}}'], ['text', '公式内文字', '\\text{说明文字}'],
      ['ℓ', '手写小写 l', '\\ell'], ['ℏ', '约化普朗克常数', '\\hbar'], ['ℑ', '虚部符号', '\\Im'], ['ℜ', '实部符号', '\\Re'],
      ['∂', '偏微分符号', '\\partial'], ['∇', 'nabla', '\\nabla'], ['∞', '无穷', '\\infty'],
      ['…', '低位省略号', '\\ldots'], ['⋯', '居中省略号', '\\cdots'], ['⋮', '竖直省略号', '\\vdots'], ['⋱', '对角省略号', '\\ddots'],
    ],
    '代数与拓扑': [
      ['G/H', '商群或商空间', 'G/H'], ['⟨S⟩', '生成子群', '\\langle S\\rangle'],
      ['◁', '正规子群', 'N\\triangleleft G'], ['⋊', '半直积', 'N\\rtimes H'], ['×', '直积', 'G\\times H'],
      ['⊕', '直和', 'M\\oplus N'], ['⊗', '张量积', 'M\\otimes_R N'], ['Hom', '同态集', '\\operatorname{Hom}(A,B)'],
      ['End', '自同态', '\\operatorname{End}(V)'], ['Aut', '自同构群', '\\operatorname{Aut}(G)'],
      ['im', '像', '\\operatorname{im}(f)'], ['coker', '余核', '\\operatorname{coker}(f)'],
      ['Spec', '素谱', '\\operatorname{Spec}(R)'], ['char', '特征', '\\operatorname{char}(K)'],
      ['deg', '次数', '\\deg(f)'], ['≅', '同构', 'A\\cong B'], ['≃', '同伦等价', 'X\\simeq Y'],
      ['πₙ', '同伦群', '\\pi_n(X,x_0)'], ['Hₙ', '同调群', 'H_n(X;R)'], ['Hⁿ', '上同调群', 'H^n(X;R)'],
      ['∂ₙ', '边界算子', '\\partial_n:C_n\\to C_{n-1}'], ['dⁿ', '上边缘算子', 'd^n:C^n\\to C^{n+1}'],
      ['exact', '短正合列', '0\\longrightarrow A\\longrightarrow B\\longrightarrow C\\longrightarrow0'],
      ['closure', '闭包', '\\overline{A}'], ['interior', '内部', 'A^{\\circ}'], ['boundary', '边界', '\\partial A'],
      ['π₁', '基本群', '\\pi_1(X,x_0)'], ['χ(X)', '欧拉示性数', '\\chi(X)=\\sum_{n\\ge0}(-1)^n\\operatorname{rank}H_n(X)'],
    ],
    '概率与统计': [
      ['E', 'Expectation', '\\E[{{selection}}]'], ['P', 'Probability', '\\Pbb({{selection}})'],
      ['Var', 'Variance', '\\Var({{selection}})'], ['Cov', 'Covariance', '\\Cov(X,Y)'],
      ['1', 'Indicator', '\\one\\{A\\}'], ['|', 'Conditional bar', '\\mid'],
      ['N', 'Normal distribution', '\\Normal(\\mu,\\sigma^2)'],
      ['Bern', 'Bernoulli distribution', '\\Ber(p)'], ['Beta', 'Beta distribution', '\\Beta(\\alpha,\\beta)'],
      ['Poi', 'Poisson distribution', '\\Poi(\\lambda)'],
      ['KL', 'KL divergence', '\\KL(P\\Vert Q)'], ['kl', 'Binary KL divergence', '\\kl(p,q)'],
      ['→p', 'Convergence in probability', '\\xrightarrow{p}'],
      ['→d', 'Convergence in distribution', '\\xrightarrow{d}'],
      ['a.s.', 'Almost surely', '\\text{a.s.}'],
    ],
    '机器学习与 Bandit': [
      ['R(T)', 'Regret', '\\Reg(T)=\\sum_{t=1}^{T}(\\mu^*-\\mu_{A_t})'],
      ['Nₐ(t)', 'Pull count', 'N_a(t)=\\sum_{s=1}^{t}\\one\\{A_s=a\\}'],
      ['μ̂', 'Empirical mean', '\\widehat\\mu_a(t)'], ['UCB', 'UCB index', '\\widehat\\mu_a(t)+\\sqrt{\\frac{2\\log t}{N_a(t)}}'],
      ['Aₜ', 'Action at time t', 'A_t'], ['Hₜ', 'History', 'H_t=(A_1,X_1,\\ldots,A_t,X_t)'],
      ['𝓕ₜ', 'Filtration', '\\calF_t'], ['Δₐ', 'Suboptimality gap', '\\Delta_a=\\mu^*-\\mu_a'],
      ['θ̂', 'Estimated parameter', '\\widehat\\theta_n'], ['ℒ', 'Loss', '\\mathcal{L}(\\theta)'],
      ['∇L', 'Gradient of loss', '\\nabla_\\theta\\mathcal{L}(\\theta)'],
      ['softmax', 'Softmax', '\\operatorname{softmax}(z)_i=\\frac{e^{z_i}}{\\sum_j e^{z_j}}'],
    ],
  };

  const elements = {
    auth: document.querySelector('#editor-auth'),
    authStatus: document.querySelector('#editor-auth-status'),
    connect: document.querySelector('#editor-connect'),
    compileCopy: document.querySelector('#editor-compile-copy'),
    compileItems: document.querySelector('#editor-compile-items'),
    compileLog: document.querySelector('#editor-compile-log'),
    compileMeta: document.querySelector('#editor-compile-meta'),
    compileSummary: document.querySelector('#editor-compile-summary'),
    content: document.querySelector('#editor-content'),
    disconnect: document.querySelector('#editor-disconnect'),
    draftBanner: document.querySelector('#editor-draft-banner'),
    draftDiscard: document.querySelector('#editor-draft-discard'),
    draftMessage: document.querySelector('#editor-draft-message'),
    draftRestore: document.querySelector('#editor-draft-restore'),
    filename: document.querySelector('#editor-filename'),
    files: document.querySelector('#editor-files'),
    formulaCategory: document.querySelector('#formula-category'),
    formulaModes: [...document.querySelectorAll('[data-formula-mode]')],
    formulaSearch: document.querySelector('#formula-search'),
    formulaSymbols: document.querySelector('#formula-symbols'),
    image: document.querySelector('#editor-image'),
    imageAlt: document.querySelector('#editor-image-alt'),
    imageControls: document.querySelector('#editor-image-controls'),
    imageDelete: document.querySelector('#editor-image-delete'),
    imageFile: document.querySelector('#editor-image-file'),
    imageReset: document.querySelector('#editor-image-reset'),
    imageWidth: document.querySelector('#editor-image-width'),
    imageWidthValue: document.querySelector('#editor-image-width-value'),
    identity: document.querySelector('#editor-identity'),
    message: document.querySelector('#editor-message'),
    markdownActions: [...document.querySelectorAll('[data-md-action]')],
    modes: [...document.querySelectorAll('.editor-modes [data-mode]')],
    newNote: document.querySelector('#editor-new'),
    panes: document.querySelector('.editor-panes'),
    preview: document.querySelector('#editor-preview'),
    previewStatus: document.querySelector('#editor-preview-status'),
    privacy: document.querySelector('.editor-privacy'),
    privacyState: document.querySelector('#editor-privacy-state'),
    privacyToggle: document.querySelector('#editor-privacy-toggle'),
    publish: document.querySelector('#editor-publish'),
    remember: document.querySelector('#editor-remember'),
    status: document.querySelector('#editor-status'),
    share: document.querySelector('#editor-share'),
    token: document.querySelector('#editor-token'),
    unshare: document.querySelector('#editor-unshare'),
    workspace: document.querySelector('#editor-workspace'),
  };

  let token = localStorage.getItem(tokenKey) || sessionStorage.getItem(tokenKey) || '';
  let currentPath = '';
  let currentSha = '';
  let currentRepository = 'public';
  let currentVisibility = 'public';
  let currentShare = null;
  let formulaMode = 'inline';
  let compileLogText = '';
  let currentCompileIssues = [];
  let markdownIssues = [];
  let pendingDraft = null;
  let previewTimer;
  let previewSyncTimer;
  let saveTimer;
  let selectedPreviewImage = null;
  let savedPreviewRange = null;
  let activeEditingSurface = 'source';
  let pendingImageTarget = 'source';
  let imageSequence = 0;
  const localImageUrls = new Map();
  const frontMatterPattern = /^---\s*\n[\s\S]*?\n---\s*(?:\n|$)/;

  const turndownService = new window.TurndownService({
    headingStyle: 'atx',
    codeBlockStyle: 'fenced',
    bulletListMarker: '-',
    emDelimiter: '_',
  });
  if (window.turndownPluginGfm?.gfm) turndownService.use(window.turndownPluginGfm.gfm);

  function htmlAttribute(value) {
    return String(value || '')
      .replaceAll('&', '&amp;')
      .replaceAll('"', '&quot;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;');
  }

  function markdownAlt(value) {
    return String(value || '').replace(/([\\\[\]])/g, '\\$1');
  }

  turndownService.addRule('mathSource', {
    filter: (node) => node.nodeType === 1 && node.hasAttribute('data-md-math-source'),
    replacement: (_content, node) => {
      try { return decodeURIComponent(node.getAttribute('data-md-math-source')); }
      catch (_) { return node.textContent; }
    },
  });

  turndownService.addRule('footnoteReference', {
    filter: (node) => node.nodeType === 1 && node.classList.contains('footnote-ref'),
    replacement: (_content, node) => {
      const id = node.querySelector('a')?.getAttribute('href')?.replace(/^#fn-/, '');
      return id ? `[^${id}]` : '';
    },
  });

  turndownService.addRule('footnoteDefinitions', {
    filter: (node) => node.nodeType === 1 && node.classList.contains('footnotes'),
    replacement: (_content, node) => {
      const definitions = [...node.querySelectorAll(':scope > ol > li')].map((item) => {
        const clone = item.cloneNode(true);
        clone.querySelectorAll('.footnote-backref').forEach((backlink) => backlink.remove());
        const id = item.id.replace(/^fn-/, '');
        return `[^${id}]: ${turndownService.turndown(clone.innerHTML).trim()}`;
      });
      return definitions.length ? `\n\n${definitions.join('\n')}` : '';
    },
  });

  turndownService.addRule('responsiveImage', {
    filter: 'img',
    replacement: (_content, node) => {
      const source = node.dataset.markdownSrc || node.getAttribute('src') || '';
      const alt = node.getAttribute('alt') || '';
      const width = Number.parseInt(node.style.width || node.dataset.width || '', 10);
      if (Number.isFinite(width) && width >= 10 && width <= 100 && (node.dataset.width || node.style.width)) {
        return `\n\n<img src="${htmlAttribute(source)}" alt="${htmlAttribute(alt)}" style="width: ${width}%;">\n\n`;
      }
      return `\n\n![${markdownAlt(alt)}](${source})\n\n`;
    },
  });

  function setStatus(message, kind = '') {
    elements.status.textContent = message;
    elements.status.dataset.kind = kind;
  }

  function setAuthStatus(message, kind = '') {
    elements.authStatus.textContent = message;
    elements.authStatus.dataset.kind = kind;
  }

  function clearStoredToken() {
    localStorage.removeItem(tokenKey);
    sessionStorage.removeItem(tokenKey);
  }

  function storeToken() {
    clearStoredToken();
    const storage = elements.remember.checked ? localStorage : sessionStorage;
    storage.setItem(tokenKey, token);
  }

  function updateCompileLog(issues) {
    const unique = issues.filter((issue, index, all) => (
      all.findIndex((item) => item.stage === issue.stage && item.line === issue.line && item.message === issue.message) === index
    ));
    currentCompileIssues = unique;
    const timestamp = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    elements.compileItems.replaceChildren();
    unique.forEach((issue) => {
      const item = document.createElement('li');
      const location = issue.line ? ` · 第 ${issue.line} 行` : '';
      item.textContent = `[${issue.stage}]${location} · ${issue.message}`;
      elements.compileItems.append(item);
    });
    elements.compileSummary.textContent = unique.length ? `${unique.length} 个问题` : '通过 · 0 个问题';
    elements.compileSummary.dataset.kind = unique.length ? 'error' : 'success';
    elements.compileMeta.textContent = `最后编译 ${timestamp} · Markdown / GFM / KaTeX`;
    compileLogText = unique.length
      ? unique.map((issue) => `[${issue.stage}]${issue.line ? ` line ${issue.line}` : ''}: ${issue.message}`).join('\n')
      : `[${timestamp}] Compilation passed with 0 issues.`;
    if (unique.length) elements.compileLog.open = true;
  }

  function apiHeaders() {
    return {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${token}`,
      'X-GitHub-Api-Version': '2022-11-28',
    };
  }

  async function githubRequest(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      headers: { ...apiHeaders(), ...(options.headers || {}) },
    });
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      let details = '';
      try {
        const payload = await response.json();
        details = payload.message || '';
      } catch (_) {}
      if (response.status === 401) message = '编辑身份已过期，请重新验证 GitHub token。';
      else if (response.status === 403) message = 'GitHub 拒绝了这次操作，请检查 token 的 Contents 读写权限。';
      else if (response.status === 409) message = '远端文章刚刚发生变化，请重新打开文章后再发布。';
      else if (response.status === 422) message = '文件名已存在或提交内容无效，请换一个文件名后重试。';
      else if (details) message = details;
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return response.status === 204 ? null : response.json();
  }

  function api(path, options = {}) {
    return githubRequest(`${apiRoot}${path}`, options);
  }

  function privateApi(path, options = {}) {
    return githubRequest(`${privateApiRoot}${path}`, options);
  }

  function repositoryApi(repository) {
    return repository === 'private' ? privateApi : api;
  }

  async function optionalContents(request) {
    try { return await request(); }
    catch (error) {
      if (error.status === 404) return null;
      throw error;
    }
  }

  function decodeBase64(value) {
    const binary = atob(value.replace(/\s/g, ''));
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    return new TextDecoder().decode(bytes);
  }

  function encodeBase64(value) {
    const bytes = new TextEncoder().encode(value);
    let binary = '';
    const chunk = 0x8000;
    for (let index = 0; index < bytes.length; index += chunk) {
      binary += String.fromCharCode(...bytes.subarray(index, index + chunk));
    }
    return btoa(binary);
  }

  function base64ToBytes(value) {
    const binary = atob(value.replace(/\s/g, ''));
    return Uint8Array.from(binary, (character) => character.charCodeAt(0));
  }

  function mimeTypeFromPath(path) {
    const extension = path.split('.').pop()?.toLowerCase();
    return ({ png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', gif: 'image/gif' })[extension] || 'application/octet-stream';
  }

  function randomId(bytes = 16) {
    const values = crypto.getRandomValues(new Uint8Array(bytes));
    return [...values].map((value) => value.toString(16).padStart(2, '0')).join('');
  }

  function replaceSource(markdown, source, replacement) {
    return markdown.split(source).join(replacement);
  }

  function documentImageSources(markdown) {
    try {
      const html = window.MarkdownPipeline.render(markdown);
      const document = new DOMParser().parseFromString(html, 'text/html');
      return [...new Set([...document.querySelectorAll('img')].map((image) => image.getAttribute('src')).filter(Boolean))];
    } catch (_) {
      return [];
    }
  }

  function prettyName(path) {
    return path.split('/').pop().replace(/\.md$/, '').split('-')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  }

  function unwrapDocumentFence(value) {
    const source = value.replaceAll('\r\n', '\n').trim();
    const wrapped = /^```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$/i.exec(source);
    return wrapped ? wrapped[1].trim() : source;
  }

  function inferDocumentTitle(markdown, filename) {
    const heading = /^#\s+(.+)$/m.exec(markdown);
    if (heading) return heading[1].replace(/[*_`]/g, '').trim();
    return prettyName(filename || 'research-note.md');
  }

  function inferDocumentSummary(markdown) {
    return markdown.split(/\n\s*\n/)
      .map((paragraph) => paragraph.replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1').replace(/[*_`>#]/g, '').trim())
      .find((paragraph) => paragraph && !/^(?:---|title:|subtitle:|summary:|date:|tags:)/i.test(paragraph))
      ?.slice(0, 220) || '';
  }

  function filenameFromTitle(title) {
    const slug = title.normalize('NFKD')
      .replace(/[\u0300-\u036f]/g, '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 72)
      .replace(/-+$/g, '');
    return `${slug || `note-${new Date().toISOString().replace(/\D/g, '').slice(0, 14)}`}.md`;
  }

  function normalizeDocument(markdown, filename) {
    let source = unwrapDocumentFence(markdown);
    if (/^---\s*\n/.test(source)) return `${source.trimEnd()}\n`;

    const today = new Date().toISOString().slice(0, 10);
    const title = inferDocumentTitle(source, filename);
    source = source.replace(/^#\s+.+\n+/, '');
    const summary = inferDocumentSummary(source);
    const frontMatter = [
      '---',
      `title: ${JSON.stringify(title)}`,
      'subtitle: ""',
      `summary: ${JSON.stringify(summary)}`,
      `description: ${JSON.stringify(summary)}`,
      `date: ${today}`,
      `lastmod: ${today}`,
      'weight: 90',
      'tags: []',
      'draft: false',
      'ShowToc: false',
      'hideMeta: true',
      '---',
      '',
    ].join('\n');
    return `${frontMatter}${source.trim()}\n`;
  }

  function wait(milliseconds) {
    return new Promise((resolve) => setTimeout(resolve, milliseconds));
  }

  async function monitorDeployment(commitSha) {
    for (let attempt = 0; attempt < 30; attempt += 1) {
      await wait(attempt < 2 ? 2500 : 4000);
      try {
        const data = await api(`/actions/runs?head_sha=${encodeURIComponent(commitSha)}&per_page=5`);
        const run = data.workflow_runs?.find((item) => item.event === 'push');
        if (!run) continue;
        if (run.status !== 'completed') {
          setStatus('已提交，网站正在更新...', 'success');
          continue;
        }
        if (run.conclusion === 'success') {
          setStatus(`已发布上线 | ${commitSha.slice(0, 7)}`, 'success');
        } else {
          setStatus('内容已提交，但网站构建失败；详情已写入编译日志。', 'error');
          updateCompileLog([{ stage: 'Deploy', line: 0, message: `GitHub Pages 构建结果：${run.conclusion || 'failure'}` }]);
        }
        return;
      } catch (_) {
        setStatus(`已提交 | ${commitSha.slice(0, 7)} · 网站正在后台更新`, 'success');
        return;
      }
    }
    setStatus(`已提交 | ${commitSha.slice(0, 7)} · 网站仍在后台更新`, 'success');
  }

  async function waitForPublishedUrl(url, attempts = 30) {
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      try {
        const separator = url.includes('?') ? '&' : '?';
        const response = await fetch(`${url}${separator}ready=${Date.now()}`, { cache: 'no-store' });
        if (response.ok) return true;
      } catch (_) {}
      await wait(3000);
    }
    return false;
  }

  function readDrafts() {
    try {
      return JSON.parse(localStorage.getItem(draftsKey) || '{}');
    } catch (_) {
      return {};
    }
  }

  function draftPath() {
    const filename = elements.filename.value.trim().toLowerCase() || 'new-note.md';
    const path = currentPath || (currentVisibility === 'private' ? `notes/${filename}` : `content/notes/${filename}`);
    return `${currentRepository}:${path}`;
  }

  function saveDraft() {
    if (!elements.content.value.trim()) return;
    const drafts = readDrafts();
    const path = draftPath();
    drafts[path] = {
      content: elements.content.value,
      filename: elements.filename.value,
      path: currentPath,
      repository: currentRepository,
      sha: currentSha,
      updatedAt: new Date().toISOString(),
    };
    localStorage.setItem(draftsKey, JSON.stringify(drafts));
    setStatus('草稿已保存在本机', 'success');
  }

  function removeDraft(path = draftPath()) {
    const drafts = readDrafts();
    delete drafts[path];
    localStorage.setItem(draftsKey, JSON.stringify(drafts));
  }

  function hideDraftOffer() {
    pendingDraft = null;
    elements.draftBanner.hidden = true;
  }

  function offerDraft(path, remoteContent) {
    const draft = readDrafts()[path];
    if (!draft || draft.content === remoteContent) {
      if (draft?.content === remoteContent) removeDraft(path);
      hideDraftOffer();
      return;
    }
    pendingDraft = { ...draft, key: path };
    const updated = new Date(draft.updatedAt).toLocaleString('zh-CN', { hour12: false });
    elements.draftMessage.textContent = `发现 ${updated} 保存的本地草稿`;
    elements.draftBanner.hidden = false;
  }

  function restoreDraft() {
    if (!pendingDraft) return;
    currentPath = pendingDraft.path;
    currentSha = pendingDraft.sha;
    currentRepository = pendingDraft.repository || 'public';
    currentVisibility = currentRepository;
    elements.filename.value = pendingDraft.filename;
    elements.filename.disabled = Boolean(currentPath);
    elements.content.value = pendingDraft.content;
    updatePrivacyUI();
    hideDraftOffer();
    renderPreview();
    setStatus('已恢复本地草稿', 'success');
  }

  function discardDraft() {
    if (!pendingDraft) return;
    removeDraft(pendingDraft.key);
    hideDraftOffer();
    setStatus('已放弃本地草稿');
  }

  function selectFile(path, repository = currentRepository) {
    elements.files.querySelectorAll('button').forEach((button) => {
      button.classList.toggle('active', button.dataset.path === path && button.dataset.repository === repository);
    });
  }

  function updatePrivacyUI() {
    const isPrivate = currentVisibility === 'private';
    elements.privacy.dataset.private = String(isPrivate);
    elements.privacyState.textContent = isPrivate ? '私密' : '公开';
    elements.privacyToggle.textContent = isPrivate ? '改为公开' : '设为私密';
    elements.share.textContent = isPrivate
      ? (currentShare ? '更新并复制分享链接' : '创建分享链接')
      : '复制分享链接';
    elements.unshare.hidden = !(isPrivate && currentShare);
  }

  async function loadShareState(filename) {
    currentShare = null;
    if (currentRepository !== 'private' || !filename) {
      updatePrivacyUI();
      return;
    }
    const file = await optionalContents(() => privateApi(`/contents/shares/${encodeURIComponent(filename)}.json?ref=${encodeURIComponent(branch)}`));
    if (file) {
      try {
        currentShare = { ...JSON.parse(decodeBase64(file.content)), metadataPath: file.path, metadataSha: file.sha };
      } catch (_) {
        updateCompileLog([{ stage: 'Privacy', line: 0, message: '私密分享状态文件无法读取。' }]);
      }
    }
    updatePrivacyUI();
  }

  async function hydratePrivateImages(markdown) {
    const sources = documentImageSources(markdown).filter((source) => source.startsWith('private-image://'));
    await Promise.all(sources.map(async (source) => {
      if (localImageUrls.has(source)) return;
      const path = source.replace('private-image://', '');
      const file = await privateApi(`/contents/${path}?ref=${encodeURIComponent(branch)}`);
      const blob = new Blob([base64ToBytes(file.content)], { type: mimeTypeFromPath(path) });
      localImageUrls.set(source, URL.createObjectURL(blob));
    }));
  }

  async function loadFiles(selectedPath = '', selectedRepository = currentRepository) {
    const [publicFiles, privateFiles] = await Promise.all([
      api(`/contents/content/notes?ref=${encodeURIComponent(branch)}`),
      optionalContents(() => privateApi(`/contents/notes?ref=${encodeURIComponent(branch)}`)),
    ]);
    const markdownFiles = [
      ...publicFiles
        .filter((file) => file.type === 'file' && file.name.endsWith('.md') && file.name !== '_index.md')
        .map((file) => ({ ...file, repository: 'public' })),
      ...(privateFiles || [])
        .filter((file) => file.type === 'file' && file.name.endsWith('.md'))
        .map((file) => ({ ...file, repository: 'private' })),
    ].sort((left, right) => left.name.localeCompare(right.name) || left.repository.localeCompare(right.repository));

    elements.files.replaceChildren();
    markdownFiles.forEach((file) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.path = file.path;
      button.dataset.repository = file.repository;
      button.textContent = prettyName(file.path);
      button.title = file.repository === 'private' ? '仅你可见' : '公开笔记';
      button.addEventListener('click', () => loadFile(file.path, file.repository));
      elements.files.append(button);
    });
    if (selectedPath) selectFile(selectedPath, selectedRepository);
    return markdownFiles;
  }

  async function loadFile(path, repository = 'public') {
    if (saveTimer) {
      clearTimeout(saveTimer);
      saveDraft();
    }
    setStatus('Loading...');
    try {
      const file = await repositoryApi(repository)(`/contents/${path}?ref=${encodeURIComponent(branch)}`);
      currentPath = path;
      currentSha = file.sha;
      currentRepository = repository;
      currentVisibility = repository;
      elements.filename.value = path.split('/').pop();
      elements.filename.disabled = true;
      const remoteContent = decodeBase64(file.content);
      const editableContent = window.MarkdownPipeline.prepareForEdit(remoteContent);
      elements.content.value = editableContent;
      await hydratePrivateImages(editableContent);
      await loadShareState(elements.filename.value);
      selectFile(path, repository);
      renderPreview();
      offerDraft(`${repository}:${path}`, editableContent);
      setStatus(repository === 'private' ? '私密笔记 · 仅你可见' : 'Ready', 'success');
    } catch (error) {
      setStatus(error.message, 'error');
    }
  }

  function newNote() {
    if (saveTimer) {
      clearTimeout(saveTimer);
      saveDraft();
    }
    currentPath = '';
    currentSha = '';
    currentRepository = 'public';
    currentVisibility = 'public';
    currentShare = null;
    elements.filename.disabled = false;
    elements.filename.value = 'new-note.md';
    elements.content.value = '# Note title\n\nStart writing here.\n';
    updatePrivacyUI();
    selectFile('', 'public');
    renderPreview();
    offerDraft(draftPath(), elements.content.value);
    elements.filename.focus();
    elements.filename.select();
    setStatus('New document', 'success');
  }

  function hideImageControls() {
    selectedPreviewImage?.classList.remove('is-selected');
    selectedPreviewImage = null;
    elements.imageControls.hidden = true;
  }

  function selectPreviewImage(image) {
    if (!image) return;
    selectedPreviewImage?.classList.remove('is-selected');
    selectedPreviewImage = image;
    selectedPreviewImage.classList.add('is-selected');
    const width = Number.parseInt(image.style.width || image.dataset.width || '100', 10);
    elements.imageWidth.value = String(Math.min(100, Math.max(10, width || 100)));
    elements.imageWidthValue.value = `${elements.imageWidth.value}%`;
    elements.imageWidthValue.textContent = `${elements.imageWidth.value}%`;
    elements.imageAlt.value = image.getAttribute('alt') || '';
    elements.imageControls.hidden = false;
  }

  function scheduleDraftSave() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveDraft, 650);
  }

  function syncPreviewToMarkdown() {
    const document = elements.preview.contentDocument;
    const body = document?.body;
    if (!body) return;
    const clone = body.cloneNode(true);
    clone.querySelectorAll('[data-upload-marker]').forEach((marker) => marker.remove());
    const frontMatter = frontMatterPattern.exec(elements.content.value)?.[0] || '';
    const markdown = turndownService.turndown(clone.innerHTML).trim();
    const nextValue = `${frontMatter}${markdown}${markdown ? '\n' : ''}`;
    if (nextValue === elements.content.value) return;
    elements.content.value = nextValue;
    markdownIssues = window.MarkdownPipeline.diagnose(nextValue);
    updateCompileLog(markdownIssues);
    elements.previewStatus.textContent = '可视化修改已同步到 Markdown';
    elements.previewStatus.dataset.kind = 'success';
    scheduleDraftSave();
  }

  function schedulePreviewSync() {
    clearTimeout(previewSyncTimer);
    previewSyncTimer = setTimeout(syncPreviewToMarkdown, 220);
  }

  function rememberPreviewRange() {
    const document = elements.preview.contentDocument;
    const selection = document?.getSelection();
    if (!selection?.rangeCount || !document.body.contains(selection.anchorNode)) return;
    savedPreviewRange = selection.getRangeAt(0).cloneRange();
    activeEditingSurface = 'preview';
  }

  function insertPreviewUploadMarker() {
    const document = elements.preview.contentDocument;
    if (!document?.body) return null;
    const marker = document.createElement('span');
    marker.dataset.uploadMarker = String(Date.now());
    marker.contentEditable = 'false';
    marker.textContent = '正在上传图片...';
    marker.className = 'image-upload-marker';
    const range = savedPreviewRange?.cloneRange();
    if (range && document.body.contains(range.commonAncestorContainer)) {
      range.collapse(false);
      range.insertNode(marker);
    } else {
      document.body.append(marker);
    }
    marker.scrollIntoView({ block: 'nearest' });
    return marker;
  }

  function applyLocalImageSources(document) {
    document.querySelectorAll('img').forEach((image) => {
      const markdownSource = image.dataset.markdownSrc || image.getAttribute('src') || '';
      image.dataset.markdownSrc = markdownSource;
      const localSource = localImageUrls.get(markdownSource);
      if (localSource) image.src = localSource;
      const width = Number.parseInt(image.style.width, 10);
      if (Number.isFinite(width)) image.dataset.width = String(width);
    });
  }

  function imageFilesFromTransfer(transfer) {
    const files = [...(transfer?.files || [])];
    const keys = new Set(files.map((file) => `${file.name}:${file.size}:${file.type}:${file.lastModified}`));
    for (const item of [...(transfer?.items || [])]) {
      if (item.kind !== 'file' || !item.type.startsWith('image/')) continue;
      const file = item.getAsFile();
      const key = file ? `${file.name}:${file.size}:${file.type}:${file.lastModified}` : '';
      if (file && !keys.has(key)) {
        keys.add(key);
        files.push(file);
      }
    }
    return files.filter((file) => file.type.startsWith('image/'));
  }

  function hasImageTransfer(transfer) {
    return imageFilesFromTransfer(transfer).length > 0
      || [...(transfer?.items || [])].some((item) => item.kind === 'file' && item.type.startsWith('image/'));
  }

  function attachPreviewInteractions(document) {
    const body = document.body;
    body.contentEditable = 'true';
    body.spellcheck = true;
    body.setAttribute('aria-label', '可直接编辑的 Markdown 预览');
    document.querySelectorAll('[data-md-math-source]').forEach((math) => {
      math.contentEditable = 'false';
      math.title = '公式请在 Markdown 源文档中编辑';
    });
    document.addEventListener('selectionchange', rememberPreviewRange);
    body.addEventListener('focus', () => { activeEditingSurface = 'preview'; });
    body.addEventListener('input', schedulePreviewSync);
    body.addEventListener('click', (event) => {
      const image = event.target.closest?.('img');
      if (image) {
        event.preventDefault();
        selectPreviewImage(image);
        return;
      }
      if (event.target.closest?.('a')) event.preventDefault();
      if (!event.target.closest?.('[data-md-math-source]')) hideImageControls();
    });
    body.addEventListener('paste', (event) => {
      const files = imageFilesFromTransfer(event.clipboardData);
      if (!files.length) return;
      event.preventDefault();
      rememberPreviewRange();
      uploadImages(files, 'preview');
    });
    body.addEventListener('dragover', (event) => {
      if (!hasImageTransfer(event.dataTransfer)) return;
      event.preventDefault();
      body.classList.add('image-drop-active');
    });
    body.addEventListener('dragleave', () => body.classList.remove('image-drop-active'));
    body.addEventListener('drop', (event) => {
      const files = imageFilesFromTransfer(event.dataTransfer);
      body.classList.remove('image-drop-active');
      if (!files.length) return;
      event.preventDefault();
      const range = document.caretRangeFromPoint?.(event.clientX, event.clientY);
      if (range) savedPreviewRange = range.cloneRange();
      activeEditingSurface = 'preview';
      uploadImages(files, 'preview');
    });
  }

  function renderPreview() {
    hideImageControls();
    markdownIssues = window.MarkdownPipeline.diagnose(elements.content.value);
    let rendered = '';
    try {
      rendered = window.MarkdownPipeline.render(elements.content.value, { preserveMathSource: true });
    } catch (error) {
      markdownIssues.push({ stage: 'Markdown', line: 0, message: error.message });
      rendered = '<p>Preview compilation failed.</p>';
    }
    updateCompileLog(markdownIssues);
    const katexBase = `${window.location.origin}/vendor/katex`;
    const contentCss = `${window.location.origin}/css/markdown-content.css`;
    elements.preview.onload = () => {
      const mathErrors = [];
      const document = elements.preview.contentDocument;
      applyLocalImageSources(document);
      try {
        window.renderMathInElement(document.body, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true },
          ],
          throwOnError: false,
          strict: false,
          errorCallback: (message) => mathErrors.push(String(message)),
          macros: mathMacros,
        });
      } catch (error) {
        mathErrors.push(error.message);
      }
      attachPreviewInteractions(document);
      updateCompileLog([
        ...markdownIssues,
        ...mathErrors.map((message) => ({ stage: 'KaTeX', line: 0, message })),
      ]);
      elements.previewStatus.textContent = mathErrors.length ? `${mathErrors.length} 个公式需要检查` : 'Markdown · GFM · KaTeX';
      elements.previewStatus.dataset.kind = mathErrors.length ? 'error' : 'success';
    };
    elements.preview.srcdoc = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="${katexBase}/katex.min.css">
<link rel="stylesheet" href="${contentCss}">
<style>
*{box-sizing:border-box}body{min-height:100%;margin:0;padding:24px;outline:0}a{color:#176d73}.markdown-content h1{font-size:34px}.markdown-content h2{font-size:28px}.markdown-content h3{font-size:22px}
.markdown-content img{cursor:pointer;transition:outline-color .15s ease}.markdown-content img.is-selected{outline:2px solid #176d73;outline-offset:4px}
.md-math-source{cursor:not-allowed}.image-upload-marker{display:inline-block;margin:8px 0;padding:5px 8px;background:#f0f5f4;color:#176d73;font:12px Arial,sans-serif}
body.image-drop-active::after{content:'松开以上传图片';position:fixed;inset:12px;display:grid;place-items:center;border:1px dashed #176d73;background:rgba(255,255,255,.94);color:#176d73;font:700 14px Arial,sans-serif;pointer-events:none}
</style></head><body>${rendered}
</body></html>`.replace('<body>', '<body class="markdown-content">');
  }

  function insertAtCursor(value) {
    const start = elements.content.selectionStart;
    const end = elements.content.selectionEnd;
    elements.content.setRangeText(value, start, end, 'end');
    elements.content.focus();
    renderPreview();
  }

  function transformSelection(action) {
    const start = elements.content.selectionStart;
    const end = elements.content.selectionEnd;
    const selected = elements.content.value.slice(start, end);
    const linePrefix = (prefix) => (selected || 'Text').split('\n').map((line, index) => `${typeof prefix === 'function' ? prefix(index) : prefix}${line}`).join('\n');
    const actions = {
      h1: () => linePrefix('# '),
      h2: () => linePrefix('## '),
      h3: () => linePrefix('### '),
      h4: () => linePrefix('#### '),
      bold: () => `**${selected || 'bold text'}**`,
      italic: () => `_${selected || 'italic text'}_`,
      strike: () => `~~${selected || 'strikethrough'}~~`,
      quote: () => linePrefix('> '),
      link: () => `[${selected || 'link text'}](https://example.com)`,
      'inline-code': () => `\`${selected || 'code'}\``,
      'code-block': () => `\n\`\`\`text\n${selected || 'code'}\n\`\`\`\n`,
      'unordered-list': () => linePrefix('- '),
      'ordered-list': () => linePrefix((index) => `${index + 1}. `),
      'task-list': () => linePrefix('- [ ] '),
      table: () => `\n| Column 1 | Column 2 | Column 3 |\n| --- | --- | --- |\n| Value | Value | Value |\n`,
      rule: () => '\n---\n',
      footnote: () => `${selected || 'Statement'}[^1]\n\n[^1]: Footnote text.`,
    };
    const replacement = actions[action]?.();
    if (replacement === undefined) return;
    elements.content.setRangeText(replacement, start, end, 'end');
    elements.content.focus();
    renderPreview();
  }

  function insertFormula(template) {
    const selected = elements.content.value.slice(elements.content.selectionStart, elements.content.selectionEnd) || 'x';
    const formula = template.replaceAll('{{selection}}', selected);
    const wrapped = formulaMode === 'display' ? `\n$$\n${formula}\n$$\n` : ` $${formula}$ `;
    insertAtCursor(wrapped);
  }

  function renderFormulaPalette(category) {
    elements.formulaSymbols.replaceChildren();
    const query = elements.formulaSearch.value.trim().toLocaleLowerCase('zh-CN');
    const entries = query
      ? Object.entries(formulaCatalog)
        .flatMap(([group, items]) => items.map((item) => [group, item]))
        .filter(([group, [label, title, template]]) => (
          `${group} ${label} ${title} ${template}`.toLocaleLowerCase('zh-CN').includes(query)
        ))
      : (formulaCatalog[category] || []).map((item) => [category, item]);
    entries.forEach(([group, [label, title, template]]) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = label;
      button.title = `${title} · ${group}`;
      button.setAttribute('aria-label', title);
      button.addEventListener('click', () => insertFormula(template));
      elements.formulaSymbols.append(button);
    });
    if (!entries.length) {
      const empty = document.createElement('p');
      empty.className = 'formula-empty';
      empty.textContent = '没有匹配的公式';
      elements.formulaSymbols.append(empty);
    }
  }

  function initializeFormulaPalette() {
    Object.keys(formulaCatalog).forEach((category) => {
      const option = document.createElement('option');
      option.value = category;
      option.textContent = category;
      elements.formulaCategory.append(option);
    });
    renderFormulaPalette(Object.keys(formulaCatalog)[0]);
  }

  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(',')[1]);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  function validateImage(file) {
    const allowedTypes = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/gif']);
    if (!allowedTypes.has(file.type)) throw new Error('图片仅支持 PNG、JPEG、WebP 或 GIF。');
    if (file.size > 8 * 1024 * 1024) throw new Error(`${file.name} 超过 8 MB，请压缩后重试。`);
  }

  async function uploadImageAsset(file, index, total) {
    validateImage(file);
    const extensions = { 'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp', 'image/gif': 'gif' };
    const extension = extensions[file.type];
    const noteSlug = (elements.filename.value || 'note').replace(/\.md$/i, '').replace(/[^a-z0-9-]/gi, '-').toLowerCase();
    imageSequence += 1;
    const assetName = `${noteSlug}-${Date.now()}-${imageSequence}.${extension}`;
    const isPrivate = currentVisibility === 'private';
    const assetPath = isPrivate
      ? `images/${noteSlug}/${assetName}`
      : `static/images/notes/uploads/${assetName}`;
    const publicPath = isPrivate
      ? `private-image://${assetPath}`
      : `/images/notes/uploads/${assetName}`;
    setStatus(`正在上传图片 ${index + 1}/${total}...`);
    const result = await repositoryApi(isPrivate ? 'private' : 'public')(`/contents/${assetPath}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: `${isPrivate ? 'Store private' : 'Upload'} figure for ${noteSlug}`,
        content: await fileToBase64(file),
        branch,
      }),
    });
    localImageUrls.set(publicPath, URL.createObjectURL(file));
    return {
      alt: file.name.replace(/\.[^.]+$/, '') || 'Research figure',
      commit: result.commit?.sha || '',
      publicPath,
    };
  }

  function createPreviewImage(document, image) {
    const element = document.createElement('img');
    element.src = localImageUrls.get(image.publicPath) || image.publicPath;
    element.dataset.markdownSrc = image.publicPath;
    element.alt = image.alt;
    return element;
  }

  async function uploadImages(files, target = 'source') {
    const images = [...files];
    if (!images.length || elements.image.disabled) return;
    try {
      images.forEach(validateImage);
    } catch (error) {
      setStatus(error.message, 'error');
      updateCompileLog([{ stage: 'Upload', line: 0, message: error.message }]);
      return;
    }

    elements.image.disabled = true;
    elements.publish.disabled = true;
    let sourceMarker = '';
    let previewMarker = null;
    if (target === 'preview') {
      previewMarker = insertPreviewUploadMarker();
    } else {
      sourceMarker = `<!-- image-upload-${Date.now()}-${imageSequence} -->`;
      elements.content.setRangeText(sourceMarker, elements.content.selectionStart, elements.content.selectionEnd, 'end');
      renderPreview();
    }

    try {
      const uploaded = [];
      for (let index = 0; index < images.length; index += 1) {
        uploaded.push(await uploadImageAsset(images[index], index, images.length));
      }
      if (target === 'preview') {
        if (previewMarker?.isConnected) {
          const document = elements.preview.contentDocument;
          const fragment = document.createDocumentFragment();
          uploaded.forEach((image) => fragment.append(createPreviewImage(document, image)));
          const lastImage = fragment.lastChild;
          previewMarker.replaceWith(fragment);
          syncPreviewToMarkdown();
          if (lastImage) selectPreviewImage(lastImage);
        } else {
          const markdown = uploaded.map((image) => `![${markdownAlt(image.alt)}](${image.publicPath})`).join('\n\n');
          elements.content.value = `${elements.content.value.trimEnd()}\n\n${markdown}\n`;
          renderPreview();
          scheduleDraftSave();
        }
      } else {
        const markdown = uploaded.map((image) => `![${markdownAlt(image.alt)}](${image.publicPath})`).join('\n\n');
        elements.content.value = elements.content.value.replace(sourceMarker, `\n${markdown}\n`);
        renderPreview();
        scheduleDraftSave();
      }
      const lastCommit = uploaded.at(-1)?.commit;
      setStatus(`已上传 ${uploaded.length} 张图片${lastCommit ? ` · ${lastCommit.slice(0, 7)}` : ''}`, 'success');
    } catch (error) {
      if (sourceMarker) elements.content.value = elements.content.value.replace(sourceMarker, '');
      previewMarker?.remove();
      renderPreview();
      setStatus(error.message, 'error');
      updateCompileLog([{ stage: 'Upload', line: 0, message: error.message }]);
    } finally {
      elements.image.disabled = false;
      elements.publish.disabled = false;
    }
  }

  async function putEncoded(apiFunction, path, encodedContent, message, sha = '') {
    const payload = { message, content: encodedContent, branch };
    if (sha) payload.sha = sha;
    return apiFunction(`/contents/${path}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }

  async function deleteRemote(apiFunction, path, sha, message) {
    return apiFunction(`/contents/${path}`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, sha, branch }),
    });
  }

  async function deleteRemoteIfExists(apiFunction, path, sha, message) {
    try { return await deleteRemote(apiFunction, path, sha, message); }
    catch (error) {
      if (error.status === 404) return null;
      throw error;
    }
  }

  async function rewriteImagesForRepository(markdown, targetRepository, filename, options = {}) {
    let rewritten = markdown;
    const assets = [];
    const sources = documentImageSources(markdown);
    const noteSlug = filename.replace(/\.md$/i, '');
    for (const source of sources) {
      if (targetRepository === 'private' && source.startsWith('/images/')) {
        const publicPath = `static${source}`;
        const file = await api(`/contents/${publicPath}?ref=${encodeURIComponent(branch)}`);
        const basename = publicPath.split('/').pop();
        const privatePath = `images/${noteSlug}/${randomId(6)}-${basename}`;
        await putEncoded(privateApi, privatePath, file.content.replace(/\s/g, ''), `Store private image for ${noteSlug}`);
        const privateSource = `private-image://${privatePath}`;
        const blob = new Blob([base64ToBytes(file.content)], { type: mimeTypeFromPath(privatePath) });
        localImageUrls.set(privateSource, URL.createObjectURL(blob));
        rewritten = replaceSource(rewritten, source, privateSource);
      }
      if (targetRepository === 'public' && source.startsWith('private-image://')) {
        const privatePath = source.replace('private-image://', '');
        const file = await privateApi(`/contents/${privatePath}?ref=${encodeURIComponent(branch)}`);
        const basename = privatePath.split('/').pop();
        const existing = (options.existingAssets || []).find((asset) => asset.source === source);
        const publicPath = existing?.path || (options.shareId
          ? `static/images/private-shares/${options.shareId}/${basename}`
          : `static/images/notes/uploads/${noteSlug}-${randomId(6)}-${basename}`);
        const result = await putEncoded(api, publicPath, file.content.replace(/\s/g, ''), `Publish image for ${noteSlug}`, existing?.sha || '');
        const publicSource = `/${publicPath.replace(/^static\//, '')}`;
        assets.push({ source, path: publicPath, sha: result.content.sha, publicSource });
        rewritten = replaceSource(rewritten, source, publicSource);
      }
    }
    return { assets, markdown: rewritten };
  }

  function setFrontMatterScalar(markdown, key, yamlValue) {
    const normalized = normalizeDocument(markdown, elements.filename.value);
    const match = frontMatterPattern.exec(normalized);
    if (!match) return normalized;
    let frontMatter = match[0].trimEnd();
    const pattern = new RegExp(`^${key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}:.*$`, 'm');
    if (pattern.test(frontMatter)) frontMatter = frontMatter.replace(pattern, `${key}: ${yamlValue}`);
    else frontMatter = frontMatter.replace(/\n---$/, `\n${key}: ${yamlValue}\n---`);
    return `${frontMatter}\n\n${normalized.slice(match[0].length).trimStart()}`;
  }

  function buildShareDocument(markdown, shareId) {
    let result = setFrontMatterScalar(markdown, 'url', `"/s/${shareId}/"`);
    result = setFrontMatterScalar(result, 'privateShare', 'true');
    result = setFrontMatterScalar(result, 'robotsNoIndex', 'true');
    result = setFrontMatterScalar(result, 'sitemap', '{ disable: true }');
    result = setFrontMatterScalar(result, 'outputs', '["HTML"]');
    return `${result.trimEnd()}\n`;
  }

  async function publish(options = {}) {
    const targetVisibility = options.targetVisibility || currentVisibility;
    const targetRepository = targetVisibility === 'private' ? 'private' : 'public';
    const shareWasDisabled = elements.share.disabled;
    const unshareWasDisabled = elements.unshare.disabled;
    elements.publish.disabled = true;
    elements.privacyToggle.disabled = true;
    elements.share.disabled = true;
    elements.unshare.disabled = true;
    try {
      let filename = elements.filename.value.trim().toLowerCase();
      if (!currentPath && filename === 'new-note.md') {
        filename = filenameFromTitle(inferDocumentTitle(elements.content.value, filename));
        elements.filename.value = filename;
      }
      if (!/^[a-z0-9][a-z0-9-]*\.md$/.test(filename)) {
        throw new Error('文件名请使用小写字母、数字和连字符，例如 my-note.md。');
      }
      const path = targetRepository === 'private' ? `notes/${filename}` : `content/notes/${filename}`;
      const today = new Date().toISOString().slice(0, 10);
      let editableContent = normalizeDocument(elements.content.value, filename);
      if (/^lastmod:/m.test(editableContent)) editableContent = editableContent.replace(/^lastmod:.*$/m, `lastmod: ${today}`);
      const rewritten = await rewriteImagesForRepository(editableContent, targetRepository, filename);
      editableContent = rewritten.markdown;
      const content = window.MarkdownPipeline.prepareForPublish(editableContent);
      const publishIssues = window.MarkdownPipeline.diagnose(content);
      const katexIssues = currentCompileIssues.filter((issue) => issue.stage === 'KaTeX');
      if (publishIssues.length || katexIssues.length) {
        updateCompileLog([...publishIssues, ...katexIssues]);
        throw new Error('请先处理编译日志中的问题。');
      }
      const source = { path: currentPath, repository: currentRepository, sha: currentSha };
      const sameLocation = source.path === path && source.repository === targetRepository;
      const message = elements.message.value.trim() || `${targetRepository === 'private' ? 'Save private' : 'Publish'} ${filename}`;
      setStatus(targetRepository === 'private' ? '正在保存到私密仓库...' : 'Publishing...');
      const result = await putEncoded(repositoryApi(targetRepository), path, encodeBase64(content), message, sameLocation ? source.sha : '');
      let publicCommit = targetRepository === 'public' ? result.commit?.sha : '';
      if (source.path && !sameLocation) {
        let deleted;
        try {
          deleted = await deleteRemote(repositoryApi(source.repository), source.path, source.sha, `Move ${filename} to ${targetVisibility}`);
        } catch (error) {
          await deleteRemoteIfExists(repositoryApi(targetRepository), path, result.content.sha, `Rollback failed move of ${filename}`);
          throw error;
        }
        if (source.repository === 'public') publicCommit = deleted.commit?.sha || publicCommit;
      }
      if (targetRepository === 'public' && currentShare) {
        await removePrivateShare({ silent: true, state: currentShare });
      }
      currentPath = result.content.path;
      currentSha = result.content.sha;
      currentRepository = targetRepository;
      currentVisibility = targetVisibility;
      if (targetRepository === 'public') currentShare = null;
      elements.filename.disabled = true;
      elements.content.value = editableContent;
      if (source.path) removeDraft(`${source.repository}:${source.path}`);
      removeDraft(`${targetRepository}:${path}`);
      hideDraftOffer();
      updatePrivacyUI();
      setStatus(targetRepository === 'private'
        ? `已保存到私密仓库 | ${result.commit.sha.slice(0, 7)} · 仅你可见`
        : `已提交 | ${result.commit.sha.slice(0, 7)} · 网站正在更新`, 'success');
      try {
        await loadFiles(currentPath, currentRepository);
      } catch (_) {
        selectFile(currentPath, currentRepository);
      }
      if (publicCommit && !options.skipMonitor) monitorDeployment(publicCommit);
      return true;
    } catch (error) {
      setStatus(error.message, 'error');
      if (!currentCompileIssues.length && error.message !== '请先处理编译日志中的问题。') {
        updateCompileLog([{ stage: 'Publish', line: 0, message: error.message }]);
      }
      return false;
    } finally {
      elements.publish.disabled = false;
      elements.privacyToggle.disabled = false;
      elements.share.disabled = shareWasDisabled;
      elements.unshare.disabled = unshareWasDisabled;
    }
  }

  async function copyShareLink(link) {
    try {
      await navigator.clipboard.writeText(link);
      setStatus(`分享链接已复制 | ${link}`, 'success');
    } catch (_) {
      setStatus(`分享链接：${link}`, 'success');
    }
  }

  async function shareCurrentNote() {
    elements.share.disabled = true;
    elements.unshare.disabled = true;
    let pendingShare = null;
    try {
      if (currentVisibility === 'public') {
        if (!currentPath || currentRepository !== 'public') {
          const published = await publish({ targetVisibility: 'public' });
          if (!published) return;
        }
        const slug = elements.filename.value.replace(/\.md$/i, '');
        await copyShareLink(`${siteUrl}/notes/${slug}/share/`);
        return;
      }

      const saved = await publish({ targetVisibility: 'private', skipMonitor: true });
      if (!saved) return;
      const shareId = currentShare?.id || randomId(16);
      const rewritten = await rewriteImagesForRepository(elements.content.value, 'public', elements.filename.value, {
        shareId,
        existingAssets: currentShare?.assets || [],
      });
      const shareDocument = window.MarkdownPipeline.prepareForPublish(buildShareDocument(rewritten.markdown, shareId));
      const contentPath = currentShare?.contentPath || `content/share/${shareId}.md`;
      const contentResult = await putEncoded(api, contentPath, encodeBase64(shareDocument), `Publish private share ${shareId}`, currentShare?.contentSha || '');
      pendingShare = { contentPath, contentSha: contentResult.content.sha, assets: rewritten.assets };
      const staleAssets = (currentShare?.assets || []).filter((asset) => (
        !rewritten.assets.some((current) => current.path === asset.path)
      ));
      for (const asset of staleAssets) {
        await deleteRemoteIfExists(api, asset.path, asset.sha, `Remove stale shared asset for ${shareId}`);
      }
      const metadata = {
        id: shareId,
        url: `${siteUrl}/s/${shareId}/`,
        contentPath,
        contentSha: contentResult.content.sha,
        assets: rewritten.assets,
        updatedAt: new Date().toISOString(),
      };
      const metadataPath = `shares/${elements.filename.value}.json`;
      const metadataResult = await putEncoded(privateApi, metadataPath, encodeBase64(`${JSON.stringify(metadata, null, 2)}\n`), `Update share state for ${elements.filename.value}`, currentShare?.metadataSha || '');
      currentShare = { ...metadata, metadataPath, metadataSha: metadataResult.content.sha };
      updatePrivacyUI();
      monitorDeployment(contentResult.commit.sha);
      setStatus('独立分享页已生成，正在等待网站部署...');
      const ready = await waitForPublishedUrl(metadata.url);
      await copyShareLink(metadata.url);
      if (!ready) {
        setStatus(`分享链接已复制 | ${metadata.url} · 页面仍在后台部署`, 'success');
      }
    } catch (error) {
      if (pendingShare) {
        await deleteRemoteIfExists(api, pendingShare.contentPath, pendingShare.contentSha, 'Rollback incomplete private share').catch(() => null);
        const cleanupAssets = [...(pendingShare.assets || []), ...(currentShare?.assets || [])];
        for (const asset of cleanupAssets) {
          await deleteRemoteIfExists(api, asset.path, asset.sha, 'Remove incomplete shared asset').catch(() => null);
        }
        if (currentShare?.metadataPath && currentShare?.metadataSha) {
          await deleteRemoteIfExists(privateApi, currentShare.metadataPath, currentShare.metadataSha, 'Remove invalid share state').catch(() => null);
        }
        currentShare = null;
        updatePrivacyUI();
      }
      setStatus(error.message, 'error');
      updateCompileLog([{ stage: 'Share', line: 0, message: error.message }]);
    } finally {
      elements.share.disabled = false;
      elements.unshare.disabled = false;
    }
  }

  async function removePrivateShare(options = {}) {
    const state = options.state || currentShare;
    if (!state) return true;
    if (!options.silent && !window.confirm('取消分享后，现有分享链接将失效。确定继续吗？')) return false;
    elements.share.disabled = true;
    elements.unshare.disabled = true;
    try {
      const contentDelete = await deleteRemoteIfExists(api, state.contentPath, state.contentSha, `Revoke private share ${state.id}`);
      let lastCommit = contentDelete?.commit?.sha || '';
      for (const asset of state.assets || []) {
        const deleted = await deleteRemoteIfExists(api, asset.path, asset.sha, `Remove shared asset for ${state.id}`);
        lastCommit = deleted?.commit?.sha || lastCommit;
      }
      await deleteRemoteIfExists(privateApi, state.metadataPath || `shares/${elements.filename.value}.json`, state.metadataSha, `Remove share state for ${state.id}`);
      if (state === currentShare) currentShare = null;
      updatePrivacyUI();
      if (!options.silent) setStatus('分享已取消，原链接将在部署完成后失效。', 'success');
      if (lastCommit && !options.silent) monitorDeployment(lastCommit);
      return true;
    } catch (error) {
      setStatus(error.message, 'error');
      updateCompileLog([{ stage: 'Share', line: 0, message: error.message }]);
      return false;
    } finally {
      elements.share.disabled = false;
      elements.unshare.disabled = false;
    }
  }

  async function connect() {
    token = elements.token.value.trim() || token;
    if (!token) {
      setAuthStatus('请输入 GitHub token。', 'error');
      return;
    }
    elements.connect.disabled = true;
    setAuthStatus('正在验证身份与仓库权限...');
    try {
      const [user, repository, privateRepository] = await Promise.all([
        githubRequest('https://api.github.com/user'),
        api(''),
        privateApi('').catch((error) => {
          if (error.status === 404) {
            throw new Error(`当前 token 无法访问私密仓库 ${privateRepo}。请在 Fine-grained token 中同时选择两个仓库，并授予 Contents 读写权限。`);
          }
          throw error;
        }),
      ]);
      if (user.login.toLowerCase() !== owner.toLowerCase()) {
        throw new Error(`当前账号 ${user.login} 没有编辑权限，仅 ${owner} 可以进入。`);
      }
      if (!repository.permissions?.push) {
        throw new Error('Token 缺少此仓库 Contents 的写入权限。');
      }
      if (!privateRepository.private || !privateRepository.permissions?.push) {
        throw new Error('私密仓库未处于 Private 状态，或 token 缺少其 Contents 读写权限。');
      }
      storeToken();
      setAuthStatus(`已验证 ${user.login}`, 'success');
      elements.identity.textContent = `${user.login} · 公开库 + 私密库已连接`;
      elements.auth.hidden = true;
      elements.workspace.hidden = false;
      const requested = params.get('path');
      const normalizedRequested = requested ? requested.replaceAll('\\', '/').replace(/^content\//, '') : '';
      const requestedPath = normalizedRequested ? `content/${normalizedRequested}` : '';
      const files = await loadFiles(requestedPath, 'public');
      if (params.has('new')) newNote();
      else if (requestedPath) await loadFile(requestedPath, 'public');
      else if (files.length) await loadFile(files[0].path, files[0].repository);
    } catch (error) {
      setAuthStatus(error.message, 'error');
      clearStoredToken();
      token = '';
    } finally {
      elements.connect.disabled = false;
    }
  }

  initializeFormulaPalette();
  elements.markdownActions.forEach((button) => button.addEventListener('click', () => transformSelection(button.dataset.mdAction)));
  elements.formulaCategory.addEventListener('change', () => {
    elements.formulaSearch.value = '';
    renderFormulaPalette(elements.formulaCategory.value);
  });
  elements.formulaSearch.addEventListener('input', () => renderFormulaPalette(elements.formulaCategory.value));
  elements.formulaModes.forEach((button) => button.addEventListener('click', () => {
    formulaMode = button.dataset.formulaMode;
    elements.formulaModes.forEach((item) => item.classList.toggle('active', item === button));
  }));
  elements.image.addEventListener('click', () => {
    pendingImageTarget = activeEditingSurface;
    elements.imageFile.click();
  });
  elements.imageFile.addEventListener('change', async () => {
    const files = [...elements.imageFile.files];
    if (files.length) await uploadImages(files, pendingImageTarget);
    elements.imageFile.value = '';
  });
  elements.imageWidth.addEventListener('input', () => {
    if (!selectedPreviewImage) return;
    const width = Number.parseInt(elements.imageWidth.value, 10);
    selectedPreviewImage.style.width = `${width}%`;
    selectedPreviewImage.dataset.width = String(width);
    elements.imageWidthValue.value = `${width}%`;
    elements.imageWidthValue.textContent = `${width}%`;
    schedulePreviewSync();
  });
  elements.imageAlt.addEventListener('input', () => {
    if (!selectedPreviewImage) return;
    selectedPreviewImage.alt = elements.imageAlt.value;
    schedulePreviewSync();
  });
  elements.imageReset.addEventListener('click', () => {
    if (!selectedPreviewImage) return;
    selectedPreviewImage.style.removeProperty('width');
    delete selectedPreviewImage.dataset.width;
    elements.imageWidth.value = '100';
    elements.imageWidthValue.value = '自动';
    elements.imageWidthValue.textContent = '自动';
    syncPreviewToMarkdown();
  });
  elements.imageDelete.addEventListener('click', () => {
    if (!selectedPreviewImage) return;
    selectedPreviewImage.remove();
    hideImageControls();
    syncPreviewToMarkdown();
  });
  elements.connect.addEventListener('click', connect);
  elements.token.addEventListener('keydown', (event) => { if (event.key === 'Enter') connect(); });
  elements.disconnect.addEventListener('click', () => { clearStoredToken(); window.location.reload(); });
  elements.compileCopy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(compileLogText);
      elements.compileCopy.textContent = '已复制';
      setTimeout(() => { elements.compileCopy.textContent = '复制日志'; }, 1200);
    } catch (_) {
      setStatus('浏览器未允许复制日志', 'error');
    }
  });
  elements.draftRestore.addEventListener('click', restoreDraft);
  elements.draftDiscard.addEventListener('click', discardDraft);
  elements.newNote.addEventListener('click', newNote);
  elements.publish.addEventListener('click', () => publish());
  elements.privacyToggle.addEventListener('click', async () => {
    const targetVisibility = currentVisibility === 'private' ? 'public' : 'private';
    const message = targetVisibility === 'private'
      ? (currentPath
        ? '这会把正文和文中本站图片迁入私密仓库，并删除公开页面。已经公开过的内容仍可能存在于 Git 历史或外部缓存中。继续吗？'
        : '这篇新笔记将只保存到私密仓库，不会进入公开网站。继续吗？')
      : '公开后，任何人都可以访问这篇笔记。确定公开吗？';
    if (!window.confirm(message)) return;
    await publish({ targetVisibility });
  });
  elements.share.addEventListener('click', shareCurrentNote);
  elements.unshare.addEventListener('click', () => removePrivateShare());
  elements.content.addEventListener('focus', () => { activeEditingSurface = 'source'; });
  elements.content.addEventListener('paste', (event) => {
    const files = imageFilesFromTransfer(event.clipboardData);
    if (!files.length) return;
    event.preventDefault();
    activeEditingSurface = 'source';
    uploadImages(files, 'source');
  });
  elements.content.addEventListener('dragover', (event) => {
    if (!hasImageTransfer(event.dataTransfer)) return;
    event.preventDefault();
    elements.content.closest('.editor-pane').classList.add('image-drop-active');
  });
  elements.content.addEventListener('dragleave', () => {
    elements.content.closest('.editor-pane').classList.remove('image-drop-active');
  });
  elements.content.addEventListener('drop', (event) => {
    const files = imageFilesFromTransfer(event.dataTransfer);
    elements.content.closest('.editor-pane').classList.remove('image-drop-active');
    if (!files.length) return;
    event.preventDefault();
    activeEditingSurface = 'source';
    uploadImages(files, 'source');
  });
  elements.content.addEventListener('input', () => {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(renderPreview, 180);
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveDraft, 650);
  });
  elements.filename.addEventListener('input', () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveDraft, 650);
  });
  elements.modes.forEach((button) => button.addEventListener('click', () => {
    elements.modes.forEach((item) => item.classList.toggle('active', item === button));
    elements.panes.dataset.mode = button.dataset.mode;
  }));
  window.addEventListener('beforeunload', () => {
    localImageUrls.forEach((url) => URL.revokeObjectURL(url));
  });

  if (token) {
    elements.remember.checked = Boolean(localStorage.getItem(tokenKey));
    elements.token.value = token;
    setAuthStatus('正在恢复此设备的编辑身份...');
    connect();
  }
})();
