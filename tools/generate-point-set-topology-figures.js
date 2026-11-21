const fs = require('fs');
const path = require('path');

const outDir = path.join(__dirname, '..', 'source', 'images', 'notes', 'assets', 'point-set-topology');
fs.mkdirSync(outDir, { recursive: true });

const escapeXml = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const line = (x1, y1, x2, y2, extra = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${extra}/>`;
const pathShape = (d, extra = '') => `<path d="${d}" ${extra}/>`;
const ellipse = (cx, cy, rx, ry, extra = '') => `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" ${extra}/>`;
const dot = (cx, cy, r = 4, extra = '') => `<circle cx="${cx}" cy="${cy}" r="${r}" ${extra}/>`;
const label = (x, y, value, extra = '') => `<text x="${x}" y="${y}" ${extra}>${escapeXml(value)}</text>`;
const plane = (x = 50, y = 245, w = 330, depth = 65, extra = '') => pathShape(`M${x} ${y}L${x + w} ${y}L${x + w + depth} ${y - depth}L${x + depth} ${y - depth}Z`, `fill="url(#fineHatch)" stroke="#171717" stroke-width="1.5" ${extra}`);
const arrow = (x1, y1, x2, y2, extra = '') => line(x1, y1, x2, y2, `stroke="#171717" stroke-width="1.6" marker-end="url(#arrow)" ${extra}`);
const cube = (x, y, w, h, dx = 46, dy = -35) =>
  pathShape(`M${x} ${y}L${x + dx} ${y + dy}L${x + dx + w} ${y + dy}L${x + w} ${y}Z`, 'fill="url(#fineHatch)" stroke="#171717" stroke-width="1.5"') +
  pathShape(`M${x + w} ${y}L${x + dx + w} ${y + dy}L${x + dx + w} ${y + dy + h}L${x + w} ${y + h}Z`, 'fill="#eee" stroke="#171717" stroke-width="1.5"') +
  pathShape(`M${x} ${y}H${x + w}V${y + h}H${x}Z`, 'fill="#fff" stroke="#171717" stroke-width="1.5"') +
  pathShape(`M${x} ${y + h}L${x + dx} ${y + dy + h}L${x + dx + w} ${y + dy + h}`, 'fill="none" stroke="#555" stroke-dasharray="5 4"');
const figures = [];
const add = (id, title, desc, body) => figures.push({ id, title, desc, body });

add('01-open-ball', '开集中的局部开球', '透视平面上的开区域包含点 x，x 周围的小圆盘完全位于区域内；边界点没有这样的圆盘。',
  plane() + pathShape('M112 203C156 157 249 155 329 184C376 202 377 230 312 235L168 235C117 234 94 218 112 203Z', 'fill="#fff" stroke="#171717" stroke-width="2"') + ellipse(226, 202, 39, 16, 'fill="url(#wideHatch)" stroke="#171717" stroke-width="1.6"') + dot(226, 202, 3) + dot(342, 225, 3, 'fill="#fff" stroke="#171717" stroke-width="1.5"') + label(237, 196, 'x') + label(350, 222, 'b') + label(150, 132, 'U open') + arrow(166, 137, 187, 177) + label(42, 269, 'X'));

add('01-basis-overlap', '基元素交中的小基元素', '两个透视椭圆形邻域相交，交中点 x 附近还有一个更小的邻域。',
  plane() + ellipse(210, 196, 98, 40, 'fill="#fff" stroke="#171717" stroke-width="1.8"') + ellipse(289, 203, 98, 40, 'fill="url(#fineHatch)" stroke="#171717" stroke-width="1.8"') + ellipse(248, 199, 31, 14, 'fill="#fff" stroke="#171717" stroke-width="2" stroke-dasharray="4 2"') + dot(248, 199, 3) + label(144, 144, 'B1') + label(344, 163, 'B2') + label(234, 147, 'B3') + arrow(247, 153, 248, 183) + label(252, 215, 'x'));

add('01-subspace', '子空间的相对开集', '大平面 X 中的带状子空间 Y 被开区域 U 截出一段在 Y 中开放、在 X 中未必开放的部分。',
  plane() + pathShape('M75 223L398 223L411 206L88 206Z', 'fill="#fff" stroke="#171717" stroke-width="1.5"') + ellipse(226, 206, 76, 36, 'fill="none" stroke="#171717" stroke-width="1.6" stroke-dasharray="5 3"') + pathShape('M150 214L301 214L294 223L143 223Z', 'fill="url(#wideHatch)" stroke="#171717" stroke-width="1"') + label(35, 266, 'X') + label(418, 219, 'Y') + label(211, 161, 'U') + label(207, 281, 'U ∩ Y') + arrow(240, 265, 237, 224));

add('02-interior-closure', '内部与闭包的三层边界', '透视平面上白色区域 A、内部点和贴着边界的闭包点；边界外的点可用邻域分开。',
  plane() + ellipse(248, 208, 110, 38, 'fill="#fff" stroke="#171717" stroke-width="2"') + ellipse(238, 204, 31, 13, 'fill="url(#wideHatch)" stroke="#171717" stroke-width="1.4"') + dot(238, 204, 3) + dot(354, 210, 4, 'fill="#fff" stroke="#171717" stroke-width="1.6"') + dot(401, 206, 4) + label(212, 151, 'interior') + arrow(239, 157, 240, 188) + label(319, 139, 'boundary') + arrow(353, 146, 354, 197) + label(380, 274, 'outside') + arrow(403, 260, 402, 215) + label(251, 236, 'A'));

add('02-continuity-preimage', '连续性用开集的原像检验', '下层平面 X 的开区域经过曲面式映射，落进上层平面 Y 的开区域 V；V 的原像在 X 中开。',
  plane(46, 264, 310, 46) + plane(91, 135, 310, 46) + ellipse(231, 226, 67, 20, 'fill="url(#wideHatch)" stroke="#171717" stroke-width="1.4"') + ellipse(269, 96, 59, 18, 'fill="url(#wideHatch)" stroke="#171717" stroke-width="1.4"') + pathShape('M231 211C242 181 256 143 268 112', 'fill="none" stroke="#171717" stroke-width="1.8" marker-end="url(#arrow)"') + label(60, 286, 'X') + label(100, 152, 'Y') + label(184, 282, 'f⁻¹(V)') + label(338, 100, 'V') + label(246, 176, 'f'));

add('02-interval-layers', '区间的内部、闭包和边界', '三层略带透视的数轴依次表示半开区间 A、开区间内部、闭区间闭包；最底层只留下两个边界点。',
  pathShape('M92 90H420L444 75H116Z M92 145H420L444 130H116Z M92 200H420L444 185H116Z M92 255H420L444 240H116Z', 'fill="url(#fineHatch)" stroke="#aaa" stroke-width="1"') +
  pathShape('M170 90H345 M170 145H345 M170 200H345', 'fill="none" stroke="#171717" stroke-width="4"') +
  [90,145,200,255].map((y,i)=>dot(170,y,5,i===1?'fill="#fff" stroke="#171717" stroke-width="1.7"':'fill="#171717"')).join('') +
  [90,145,200,255].map((y,i)=>dot(345,y,5,i<2?'fill="#fff" stroke="#171717" stroke-width="1.7"':'fill="#171717"')).join('') +
  label(18,95,'A = [0,1)') + label(18,150,'Int A') + label(18,205,'Cl A') + label(18,260,'∂A') + label(160,282,'0') + label(340,282,'1'));

add('03-product-box', '积拓扑的小长方体', '两个坐标方向的开区间相乘，得到积空间中的开矩形；第三方向表示另一坐标限制。',
  cube(118,112,190,120,62,-42) + pathShape('M158 128L348 128L370 112 M158 128L158 209 M348 128L348 209', 'fill="none" stroke="#555" stroke-dasharray="5 4"') + label(108,265,'U in X') + label(322,260,'V in Y') + arrow(163,254,193,217) + arrow(373,251,329,221) + label(209,99,'U × V'));

add('03-finite-coordinates', '无限积的有限坐标限制', '一列透视坐标平面中只有前三层被剖面线标出，其余坐标不受限制；方框拓扑会同时限制所有层。',
  [0,1,2,3,4].map((i)=>plane(90+i*26,252-i*31,230,42,i<3?'':'stroke-dasharray="5 4"')).join('') +
  [0,1,2].map(i=>ellipse(240+i*26,227-i*31,31,10,'fill="url(#wideHatch)" stroke="#171717"')).join('') +
  label(55,269,'x₁') + label(90,235,'x₂') + label(118,204,'x₃') + label(343,105,'...') + label(115,291,'only finitely many coordinates'));

add('03-quotient-cylinder', '方形的边粘成圆柱', '左侧透视方形标出两条待识别边，箭头指向右侧圆柱；圆柱背面的边用虚线表示。',
  plane(28,235,170,45) + pathShape('M72 190L28 235 M242 190L198 235', 'fill="none" stroke="#171717" stroke-width="4"') + label(46,173,'0') + label(232,174,'1') + arrow(249,188,286,188) + ellipse(372,147,64,24,'fill="url(#fineHatch)" stroke="#171717" stroke-width="1.7"') + pathShape('M308 147V233 M436 147V233', 'fill="none" stroke="#171717" stroke-width="1.7"') + pathShape('M308 233C308 246 336 257 372 257S436 246 436 233', 'fill="none" stroke="#171717" stroke-width="1.7"') + pathShape('M308 233C308 220 336 209 372 209S436 220 436 233', 'fill="none" stroke="#777" stroke-dasharray="5 4"') + label(337,286,'S¹ × I'));

add('04-nested-neighborhoods', '可数邻域基逐层缩小', '透视平面上的三个嵌套椭圆邻域 U1、U2、U3 包围同一个点 x，序列点逐层逼近 x。',
  plane() + ellipse(244,203,132,42,'fill="#fff" stroke="#171717" stroke-width="1.5"') + ellipse(244,203,89,28,'fill="url(#fineHatch)" stroke="#171717" stroke-width="1.5"') + ellipse(244,203,42,14,'fill="#fff" stroke="#171717" stroke-width="1.5"') + dot(244,203,3) + dot(145,206,3) + dot(184,201,3) + dot(223,202,3) + label(112,147,'U₁') + label(156,158,'U₂') + label(217,176,'U₃') + label(251,212,'x') + label(123,226,'a₁') + label(176,227,'a₂') + label(215,225,'a₃'));

add('04-sequence-vs-net', '序列与网的索引形状', '左侧线性序列沿单方向逼近极限；右侧分支有向集通过共同上界汇合到同一点。',
  plane(22,252,185,46) + plane(264,252,150,46) + pathShape('M65 218L178 218', 'fill="none" stroke="#171717" marker-end="url(#arrow)"') + [76,104,130,151,169].map((x,i)=>dot(x,218,3)).join('') + dot(184,218,5,'fill="#fff" stroke="#171717"') + label(70,172,'sequence') + label(300,172,'directed net') + pathShape('M303 236L339 208L385 222L414 190 M303 198L339 208 M385 222L414 190', 'fill="none" stroke="#171717" stroke-width="1.5" marker-end="url(#arrow)"') + [303,339,385,414].map((x,i)=>dot(x,[236,208,222,190][i],3)).join(''));

add('04-cocountable', '序列看不见的闭包点', '透视平面上点 x 被许多点围绕；一个只列出可数点的序列可被包含 x 的开集整体避开。',
  plane() + dot(250,202,5,'fill="#fff" stroke="#171717" stroke-width="1.8"') + [148,181,218,294,331,365].map((x,i)=>{const y=[198,214,192,226,186,212][i];return line(x-5,y-5,x+5,y+5,'stroke="#171717" stroke-width="1.4"')+line(x-5,y+5,x+5,y-5,'stroke="#171717" stroke-width="1.4"')}).join('') + label(249,183,'x') + label(151,151,'A = X without x') + label(112,279,'U = X without all sequence values'));

add('05-hausdorff', 'Hausdorff 空间可分开不同点', '透视平面上两个点各有一个开邻域，邻域之间留有明确间隙；虚线连接表示两点本身并不需要有距离函数。',
  plane() + ellipse(171,205,59,25,'fill="#fff" stroke="#171717" stroke-width="1.7"') + ellipse(334,205,59,25,'fill="#fff" stroke="#171717" stroke-width="1.7"') + dot(171,205,4) + dot(334,205,4) + pathShape('M230 205H275','fill="none" stroke="#777" stroke-dasharray="4 4"') + label(165,168,'U') + label(329,168,'V') + label(164,234,'x') + label(327,234,'y') + label(195,275,'U ∩ V = ∅'));

add('05-regular-normal', '点与闭集的分离', '左侧孤立点 x 有圆形开邻域，右侧闭集 F 有带状开邻域，两个邻域在透视平面上不相交。',
  plane() + ellipse(165,209,53,22,'fill="#fff" stroke="#171717" stroke-width="1.7"') + ellipse(333,205,80,31,'fill="#fff" stroke="#171717" stroke-width="1.7"') + ellipse(333,205,49,15,'fill="url(#wideHatch)" stroke="#171717" stroke-width="1"') + dot(165,209,4) + label(155,171,'x') + label(329,211,'F') + label(118,270,'open around x') + label(288,270,'open around F'));

add('05-urysohn-surface', '连续函数分开闭集', '透视平面上两个闭集 A、B 位于不同高度，连续曲面从函数值零平滑升到一，表示 Urysohn 函数。',
  plane(45,257,350,59) + pathShape('M91 196C158 196 186 190 220 150S300 90 368 90L418 142C350 141 315 171 269 205S151 249 91 249Z','fill="url(#fineHatch)" stroke="#171717" stroke-width="1.7"') + pathShape('M91 196C158 196 186 190 220 150S300 90 368 90','fill="none" stroke="#171717" stroke-width="2"') + pathShape('M115 195C171 194 198 184 232 147S301 91 368 91 M133 210C184 206 215 184 245 151S322 108 390 111','fill="none" stroke="#777" stroke-width="1"') + ellipse(119,231,27,9,'fill="none" stroke="#171717"') + ellipse(381,175,27,9,'fill="none" stroke="#171717"') + line(381,175,381,92,'stroke="#555" stroke-dasharray="5 3"') + label(110,280,'A: h=0') + label(364,198,'B') + label(389,91,'h=1'));

add('06-open-cover', '紧空间的有限子覆盖', '球面被多块开片覆盖，其中三块用较深剖面线强调，表示从原开覆盖中选择有限多个即可覆盖整体。',
  '<defs><clipPath id="sphere"><ellipse cx="245" cy="190" rx="123" ry="91"/></clipPath></defs>' + ellipse(245,190,123,91,'fill="#fff" stroke="#171717" stroke-width="1.8"') + '<g clip-path="url(#sphere)">' + ellipse(178,188,91,100,'fill="url(#fineHatch)" stroke="#171717" stroke-width="1"') + ellipse(247,181,92,102,'fill="url(#fineHatch)" stroke="#171717" stroke-width="1"') + ellipse(328,195,88,102,'fill="url(#fineHatch)" stroke="#171717" stroke-width="1"') + '</g>' + ellipse(245,190,123,91,'fill="none" stroke="#171717" stroke-width="1.8"') + label(147,86,'U₁') + label(243,77,'U₂') + label(343,101,'U₃') + label(185,288,'three selected open patches'));

add('06-finite-intersection', '闭集有限交性质', '透视平面上逐层嵌套的三个闭区域保留一个共同点；交中点体现有限条件相容时紧性的全局结论。',
  plane() + ellipse(245,204,127,47,'fill="#fff" stroke="#171717" stroke-width="1.7"') + ellipse(248,203,91,34,'fill="url(#fineHatch)" stroke="#171717" stroke-width="1.5"') + ellipse(251,202,52,20,'fill="#fff" stroke="#171717" stroke-width="1.5"') + dot(251,203,4) + label(113,149,'F₁') + label(158,157,'F₂') + label(206,173,'F₃') + label(259,209,'x') + label(135,276,'the common intersection'));

add('06-compact-product', '有限积中的小长方体', '透视长方体表示积空间，若干小长方体片覆盖其边面，示意有限积紧性的管状邻域证明。',
  cube(114,105,198,128,70,-45) + pathShape('M149 105V233 M199 105V233 M249 105V233 M114 150H312 M114 192H312 M350 60V188','fill="none" stroke="#777" stroke-width="1"') + pathShape('M199 105V150H249V105 M249 150V192H312','fill="url(#wideHatch)" stroke="#171717" stroke-width="1.5"') + label(181,92,'X × Y') + label(75,260,'finite slices along X') + label(320,256,'Y'));

add('06-total-bounded', '全有界的有限小球覆盖', '透视平面上的一块区域由有限个同半径的小球覆盖；每个半径都须能重新找到这样的有限覆盖。',
  plane() + pathShape('M123 205H353','fill="none" stroke="#171717" stroke-width="3"') + [145,207,269,331].map((x,i)=>ellipse(x,204+(i%2)*5,54,22,'fill="none" stroke="#171717" stroke-width="1.4"')).join('') + [145,207,269,331].map((x,i)=>dot(x,204+(i%2)*5,3)).join('') + label(126,146,'ε-balls') + label(102,246,'X') + label(137,280,'finite cover at each scale'));

add('07-disconnected', '两个开闭分支', '透视平面上两个相隔的实心区域各自可由开邻域包围；它们组成不连通空间的两个开闭部分。',
  plane() + ellipse(170,201,67,26,'fill="url(#wideHatch)" stroke="#171717" stroke-width="1.7"') + ellipse(336,204,67,26,'fill="url(#wideHatch)" stroke="#171717" stroke-width="1.7"') + label(159,207,'U') + label(330,210,'V') + label(182,275,'U ∩ V = ∅'));

add('07-intermediate-value', '连续曲线穿过中间高度', '透视坐标平面上连续曲线从低处走向高处，中间高度的虚线与曲线相交，对应中间值定理。',
  plane(60,257,340,55) + line(107,228,107,69,'stroke="#171717" stroke-width="1.4" marker-end="url(#arrow)"') + line(107,228,425,228,'stroke="#171717" stroke-width="1.4" marker-end="url(#arrow)"') + pathShape('M114 215C174 215 173 142 224 170S293 115 350 88S393 100 414 75','fill="none" stroke="#171717" stroke-width="2.5"') + line(109,150,406,150,'stroke="#777" stroke-dasharray="5 4"') + dot(264,150,4) + label(75,155,'t') + label(257,140,'f(c)=t') + label(401,246,'x'));

add('07-sine-curve', '连通而不道路连通的正弦曲线闭包', '竖直极限线位于左侧，向左无限振荡的曲线从右侧逼近；整体闭包连通，但无法用路径从竖线走到曲线上。',
  plane() + line(185,166,185,238,'stroke="#171717" stroke-width="2"') + pathShape(Array.from({length:260},(_,i)=>{const t=0.012+i*0.0038;return `${i?'L':'M'}${(185+190*t).toFixed(1)} ${(202-27*Math.sin(1/t)).toFixed(1)}`}).join(''),'fill="none" stroke="#171717" stroke-width="1.7"') + label(138,147,'limit segment') + label(290,168,'sin(1/x)') + label(116,276,'oscillation accumulates at x=0'));

add('08-cauchy-hole', 'Cauchy 序列趋向缺失的点', '透视平面上序列点沿一条轨迹靠近一个空心缺失点，点彼此越来越近却无空间内极限。',
  plane() + pathShape('M120 226C172 208 214 203 282 199','fill="none" stroke="#171717" stroke-width="1.3" stroke-dasharray="4 3"') + [130,170,202,225,242,254,263,270].map((x,i)=>dot(x,225-25*(1-Math.exp(-(x-130)/45)),Math.max(2,4-i*0.25))).join('') + dot(282,199,7,'fill="#fff" stroke="#171717" stroke-width="1.8"') + label(105,163,'x₁, x₂, ...') + label(283,173,'missing limit') + arrow(306,177,286,190) + label(150,276,'terms approach each other'));

add('08-completion', '完备化补上缺失极限', '下层透视数轴有一个空心缺口，上层完成后的数轴把该点填实；箭头表示稠密的等距嵌入。',
  plane(45,249,347,45) + line(118,213,397,213,'stroke="#171717" stroke-width="2"') + dot(270,213,6,'fill="#fff" stroke="#171717" stroke-width="2"') + label(115,282,'X: Cauchy gap') + arrow(276,193,276,132) + plane(81,130,318,37) + line(137,111,389,111,'stroke="#171717" stroke-width="2"') + dot(276,111,6) + label(113,79,'completion X̂') + label(312,86,'new point') + arrow(321,93,283,107));

add('08-baire-balls', 'Baire 定理中的嵌套闭球', '三维透视平面上越来越小的闭球嵌套，每层与一个稠密开集相容，最终保留公共极限点。',
  plane() + ellipse(247,202,128,45,'fill="#fff" stroke="#171717" stroke-width="1.7"') + ellipse(258,202,85,30,'fill="url(#fineHatch)" stroke="#171717" stroke-width="1.5"') + ellipse(264,202,44,15,'fill="#fff" stroke="#171717" stroke-width="1.5"') + dot(267,202,4) + label(115,143,'K₁') + label(178,161,'K₂') + label(227,180,'K₃') + label(272,213,'x') + label(150,273,'diameters tend to zero'));

add('08-total-complete', '全有界抽子列、完备性给极限', '三层透视区域的半径逐层缩小；序列子列始终留在越来越小的区域里，最后收敛于同一极限点。',
  plane(62,255,320,53) + ellipse(230,215,95,32,'fill="#fff" stroke="#171717"') + ellipse(245,209,57,19,'fill="url(#fineHatch)" stroke="#171717"') + ellipse(255,205,26,9,'fill="#fff" stroke="#171717"') + [175,211,237,252].map((x,i)=>dot(x,[220,215,209,205][i],3)).join('') + dot(261,204,5,'fill="#fff" stroke="#171717" stroke-width="1.5"') + label(130,150,'ε₁') + label(189,164,'ε₂') + label(230,179,'ε₃') + label(297,181,'limit') + arrow(304,184,270,201) + label(103,281,'finite covers select a Cauchy subsequence'));

add('09-urysohn-embedding', '连续函数把空间嵌入可数积', '左侧原空间的点通过三条坐标函数箭头映入右侧透视立方体；函数值同时区分点和邻域。',
  plane(20,246,145,43) + ellipse(112,217,47,17,'fill="#fff" stroke="#171717" stroke-width="1.5"') + dot(111,217,4) + cube(275,107,116,105,39,-33) + dot(342,165,5) + pathShape('M163 208C216 194 226 181 316 166','fill="none" stroke="#171717" stroke-width="1.7" marker-end="url(#arrow)"') + label(44,275,'X') + label(210,176,'F=(f₁,f₂,...)') + label(282,242,'[0,1]ᴺ'));

add('09-compact-open', '紧开拓扑控制整个紧集的像', '下层空间 X 的紧区间 K 经过映射后，所有像点都落在上层空间 Y 的开区域 U 内。',
  plane(56,256,320,54) + plane(93,135,305,43) + pathShape('M146 215H313','fill="none" stroke="#171717" stroke-width="5"') + ellipse(264,99,91,25,'fill="url(#wideHatch)" stroke="#171717" stroke-width="1.5"') + pathShape('M156 210C171 170 196 127 218 99 M304 210C296 165 318 131 332 103','fill="none" stroke="#171717" stroke-width="1.3" marker-end="url(#arrow)"') + dot(218,99,3) + dot(332,103,3) + label(218,236,'K') + label(359,104,'U') + label(74,278,'X') + label(112,155,'Y'));

add('09-evaluation', '评价映射读出函数在一点的值', '三维透视曲面表示函数图像，底面上选一点 x，垂直虚线升到图像值 f(x)。',
  plane(62,252,330,55) + pathShape('M107 190C180 176 225 140 295 135S374 155 412 120L425 172C360 202 274 190 210 233L107 243Z','fill="url(#fineHatch)" stroke="#171717" stroke-width="1.5"') + pathShape('M107 190C180 176 225 140 295 135S374 155 412 120','fill="none" stroke="#171717" stroke-width="2"') + dot(244,225,4) + line(244,225,244,153,'stroke="#171717" stroke-width="1.3" stroke-dasharray="4 3"') + dot(244,153,4) + label(234,254,'x') + label(249,151,'f(x)') + label(153,111,'graph of f'));

add('10-local-compact', '非紧空间中的紧邻域', '无界透视平面 X 中只有点 x 周围一块闭盘被圈出；整体无界而局部闭包紧。',
  plane(15,266,375,68) + ellipse(242,212,68,23,'fill="#fff" stroke="#171717" stroke-width="1.6"') + ellipse(242,212,45,15,'fill="url(#wideHatch)" stroke="#171717" stroke-width="1.4"') + dot(242,212,4) + label(246,203,'x') + label(160,166,'compact closure') + arrow(213,171,225,189) + label(18,291,'noncompact X extends beyond this view'));

add('10-one-point', '一点紧化的立体投影视角', '左侧透视平面上的远处方向通过曲线箭头汇到右侧球面的北极点 infinity，表示一点紧化。',
  plane(9,246,196,38) + pathShape('M83 229C115 214 157 206 193 199','fill="none" stroke="#171717" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrow)"') + arrow(232,179,281,160) + ellipse(369,178,76,76,'fill="#fff" stroke="#171717" stroke-width="1.8"') + pathShape('M293 178C293 197 327 210 369 210S445 197 445 178','fill="none" stroke="#777" stroke-dasharray="4 3"') + pathShape('M293 178C293 159 327 146 369 146S445 159 445 178','fill="none" stroke="#171717" stroke-width="1.1"') + dot(369,102,5) + label(379,104,'∞') + label(42,277,'R²') + label(345,274,'S²'));

add('10-partition-unity', '局部有限的单位分解', '透视底面上的两个重叠开片各有一座函数小丘，两座小丘在重叠处互补，点处函数值之和为一。',
  plane(46,260,344,60) + pathShape('M118 219H393','fill="none" stroke="#171717" stroke-width="1.5"') + pathShape('M118 113H178L324 219H393','fill="none" stroke="#171717" stroke-width="2"') + pathShape('M118 219H178L324 113H393','fill="none" stroke="#171717" stroke-width="2" stroke-dasharray="7 3"') + line(178,219,178,113,'stroke="#777" stroke-dasharray="4 3"') + line(324,219,324,113,'stroke="#777" stroke-dasharray="4 3"') + label(132,103,'φ₁=1') + label(344,103,'φ₂=1') + label(196,159,'φ₁+φ₂=1') + label(204,278,'overlap region'));

add('10-locally-finite', '局部有限覆盖的邻域判据', '透视平面上的开片沿一条方向排列，固定点 x 附近的小邻域只与两块开片相交，远处其他开片不参与局部计算。',
  plane() + [139,205,271,337].map((x,i)=>ellipse(x,204+(i%2)*5,57,21,'fill="none" stroke="#171717" stroke-width="1.4"')).join('') + dot(237,206,4) + ellipse(237,206,34,14,'fill="url(#wideHatch)" stroke="#171717" stroke-dasharray="4 3"') + label(225,174,'x') + label(153,276,'only nearby patches meet x'));

for (const { id, title, desc, body } of figures) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 40 480 260" role="img" aria-labelledby="title desc">
<title id="title">${escapeXml(title)}</title><desc id="desc">${escapeXml(desc)}</desc>
<defs><pattern id="fineHatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(40)"><path d="M0 0V6" stroke="#d0d0d0" stroke-width="1"/></pattern><pattern id="wideHatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(40)"><path d="M0 0V7" stroke="#8b8b8b" stroke-width="1.2"/></pattern><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="#171717"/></marker></defs>
<rect width="480" height="300" fill="#fff"/><g fill="#171717" font-family="Georgia, 'Times New Roman', serif" font-size="17" stroke-linecap="round" stroke-linejoin="round">${body}</g>
</svg>`;
  fs.writeFileSync(path.join(outDir, `${id}.svg`), svg);
}
console.log(`Generated ${figures.length} topology figures in ${outDir}`);
