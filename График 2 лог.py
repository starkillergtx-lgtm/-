import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from scipy.interpolate import PchipInterpolator
from google.colab import files
import io, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── 1. Загрузка CSV ──────────────────────────────────────────
print("Загрузите qoe_ultimate.csv:")
uploaded = files.upload()
frames = []

for fname, content in uploaded.items():
    tmp = pd.read_csv(io.BytesIO(content), encoding='utf-8-sig')
    frames.append(tmp)
    print(f"  ✓ {fname}: {len(tmp)} строк")

df_all = pd.concat(frames, ignore_index=True)

# ── 2. Фильтрация по Статус ──────────────────────────────────
if 'Статус' in df_all.columns:
    df = df_all[df_all['Статус'] == 'OK'].copy()
    excl = len(df_all) - len(df)
    print(f"Фильтр по Статус='OK': {len(df)} строк включено, {excl} исключено")
else:
    def tc_sec_f(tc):
        try:
            if str(tc) == 'None':
                return np.nan
            p = str(tc).split(':')
            return int(p[0]) * 60 + int(p[1])
        except:
            return np.nan

    df_all['_tc'] = df_all['Таймкод'].apply(tc_sec_f)
    df = df_all[~((df_all['MOS'] == 5) & (df_all['_tc'] > 30))].copy()
    print(f"Стандартный фильтр: {len(df)} строк")

# ── 3. Предобработка данных ──────────────────────────────────
def get_dim(res, idx):
    try:
        res_clean = str(res).replace('x', '×')
        return int(res_clean.split('×')[idx])
    except:
        return np.nan

df['width'] = df['Разрешение'].apply(lambda r: get_dim(r, 0))
df['height'] = df['Разрешение'].apply(lambda r: get_dim(r, 1))

MOS_SCORES = [5, 4, 3, 2]
RAW_FORMATS = ['4K', '1080', '720']
PLOT_FORMATS = ['4K', '1080', '720', 'Main']

COLORS = {
    '4K': '#1A6FBF',
    '1080': '#1E8A3A',
    '720': '#CC2020',
    'Main': '#111111'
}

LINE_STYLES = {
    '4K': '-',
    '1080': '--',
    '720': '-.',
    'Main': '-'
}

MOS_LABELS = {
    5: '5\n(отлично)',
    4: '4\n(хорошо)',
    3: '3\n(удовл.)',
    2: '2\n(плохо)'
}

ITU_LEVELS = [
    (4096, '4096×2160  (DCI 4K — исходное видео)'),
    (3840, '3840×2160  (UHD 4K — формат экрана)'),
    (2560, '2560×1440  (QHD / 2K)'),
    (1920, '1920×1080  (FHD)'),
    (1280, '1280×720   (HD 720p)'),
    (854,  ' 854×480   (SD 480p)'),
    (640,  ' 640×360   (360p)'),
    (426,  ' 426×240   (240p)'),
    (240,  ' 240 px    (нижняя граница лог. шкалы)')
]

LOG_TICKS = [240, 360, 480, 720, 1080, 1440, 2160, 3840, 4096, 5000]

# ── 4. Расчет статистики, ДИ и усредненной кривой ────────────
# Средние значения считаются по очищенным наблюдениям.
# Доверительные интервалы считаются консервативно:
# сначала усредняются повторы внутри каждого участника,
# затем 95% ДИ рассчитывается по межсубъектному разбросу.

S = {}

FORMAT_COL = 'Formatted_screen_size' if 'Formatted_screen_size' in df.columns else 'Формат_экрана'
SUBJECT_COL = 'Испытуемый'

for fmt in RAW_FORMATS:
    S[fmt] = {}

    for mos in MOS_SCORES:
        mask = (df[FORMAT_COL] == fmt) & (df['MOS'] == mos)
        group = df.loc[mask, [SUBJECT_COL, 'width', 'height']].dropna()

        if len(group) == 0:
            mw, mh, ci, n_obs, n_subj = np.nan, np.nan, 0.0, 0, 0
        else:
            # Среднее значение точки: по всем очищенным наблюдениям
            mw = group['width'].mean()
            mh = group['height'].mean()
            n_obs = len(group)

            # Для доверительного интервала:
            # сначала считаем среднее значение по каждому участнику
            subj_w = group.groupby(SUBJECT_COL)['width'].mean().dropna()
            n_subj = len(subj_w)

            if n_subj > 1:
                ci = stats.t.ppf(0.975, df=n_subj - 1) * stats.sem(subj_w)
            else:
                ci = 0.0

        S[fmt][mos] = {
            'mean': mw,
            'height': mh,
            'ci': ci,
            'n': n_obs,
            'n_subj': n_subj,
            'label': f'{int(round(mw))}×{int(round(mh))}' if not np.isnan(mw) else '—'
        }

# Усредненная кривая Main строится как среднее трех форматных кривых.
# ДИ для Main на этом графике не отображается, чтобы не перегружать рисунок.
S['Main'] = {}

for mos in MOS_SCORES:
    vals = [
        S[fmt][mos]['mean']
        for fmt in RAW_FORMATS
        if not np.isnan(S[fmt][mos]['mean'])
    ]

    if len(vals) > 0:
        mean_val = np.mean(vals)

        S['Main'][mos] = {
            'mean': mean_val,
            'ci': 0.0,
            'label': f'{int(round(mean_val))}'
        }
    else:
        S['Main'][mos] = {
            'mean': np.nan,
            'ci': 0.0,
            'label': '—'
        }

# ── 5. Оформление координатной сетки ─────────────────────────
def apply_grid(ax):
    ax.set_facecolor('#FDFDFD')
    ax.grid(True, which='major', color='#E2E2E2', lw=0.9, ls='-', zorder=0)
    ax.grid(True, which='minor', color='#EEEEEE', lw=0.5, ls='-', zorder=0)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for sp in ['left', 'bottom']:
        ax.spines[sp].set_linewidth(1.2)
        ax.spines[sp].set_color('#444444')

def apply_log_y_ticks(ax, y_bot, y_top):
    visible_ticks = [t for t in LOG_TICKS if y_bot <= t <= y_top]

    ax.yaxis.set_major_locator(mticker.FixedLocator(visible_ticks))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{int(v):,}' if y_bot <= v <= y_top else '')
    )

    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    )

# ── 6. Эталонные линии ITU ───────────────────────────────────
def draw_itu(ax, y_bot, y_top):
    x_pos = 1.28

    for res, label in ITU_LEVELS:
        if y_bot < res < y_top:

            # Верхние стандарты 4096 и 3840 находятся близко,
            # поэтому линии делаем светлее, а подписи разводим по вертикали.
            if res == 4096:
                line_alpha = 0.45
                text_y = res * 1.035
                text_va = 'bottom'
            elif res == 3840:
                line_alpha = 0.45
                text_y = res * 0.965
                text_va = 'top'
            else:
                line_alpha = 0.75
                text_y = res * 1.015
                text_va = 'bottom'

            ax.axhline(
                res,
                color='#BBBBBB',
                lw=1.0,
                ls='-.',
                zorder=1,
                alpha=line_alpha
            )

            ax.text(
                x_pos,
                text_y,
                label,
                ha='right',
                va=text_va,
                fontsize=8.5,
                color='#777777',
                style='italic',
                zorder=3
            )

# ── Ручное позиционирование ──────────────────────────────────
CUSTOM_LABEL_POSITIONS = {}

# ── 7. Построение финального сводного графика ────────────────
fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')

y_all = [
    S[fmt][m]['mean']
    for fmt in PLOT_FORMATS
    for m in MOS_SCORES
    if not np.isnan(S[fmt][m]['mean'])
]

Y_BOT = 150
Y_TOP = int(np.ceil(max(y_all) * 1.35 / 500.0) * 500)

X_R, X_L = 5.55, 1.25

ax.set_xlim(X_R, X_L)
ax.set_ylim(Y_BOT, Y_TOP)
ax.set_yscale('log')

apply_grid(ax)
draw_itu(ax, Y_BOT, Y_TOP)

LANE_HALF_WIDTH = 0.22

for mos in MOS_SCORES:
    ax.axvspan(
        mos - LANE_HALF_WIDTH,
        mos + LANE_HALF_WIDTH,
        facecolor='#F0F0F0',
        edgecolor='#D5D5D5',
        lw=1.0,
        alpha=0.75,
        zorder=1
    )
    ax.axvline(mos, color='#999999', lw=1.2, ls=':', zorder=1)

OFFSETS = {
    '4K': -0.18,
    '1080': -0.06,
    '720': 0.06,
    'Main': 0.18
}

# Тонкие горизонтальные проекции форматных кривых
for fmt in RAW_FORMATS:
    color = COLORS[fmt]

    for mos in MOS_SCORES:
        xi = float(mos) + OFFSETS[fmt]
        yi = S[fmt][mos]['mean']

        if np.isnan(yi):
            continue

        ax.hlines(
            yi,
            xi,
            X_R,
            colors=color,
            linestyles=':',
            lw=1.0,
            alpha=0.4,
            zorder=2
        )

for fmt in PLOT_FORMATS:
    is_main = (fmt == 'Main')
    color = COLORS[fmt]
    ls_style = LINE_STYLES[fmt]

    xv = np.array(MOS_SCORES, dtype=float) + OFFSETS[fmt]
    yv = np.array([S[fmt][m]['mean'] for m in MOS_SCORES])
    cv = np.array([S[fmt][m]['ci'] for m in MOS_SCORES])

    cs = PchipInterpolator(xv[::-1], yv[::-1])
    xd = np.linspace(xv[-1], xv[0], 400)

    line_w = 3.5 if is_main else 2.0
    line_alpha = 1.0 if is_main else 0.75
    z_line = 9 if is_main else 3

    label_name = 'Основная лестница QoE' if is_main else f'Формат: {fmt}'

    ax.plot(
        xd,
        cs(xd),
        color=color,
        ls=ls_style,
        lw=line_w,
        alpha=line_alpha,
        solid_capstyle='round',
        zorder=z_line,
        label=label_name
    )

    # ДИ показываем только для исходных форматных кривых.
    if not is_main:
        ax.errorbar(
            xv,
            yv,
            yerr=cv,
            fmt='none',
            ecolor=color,
            alpha=0.75,
            elinewidth=2.0,
            capsize=5,
            capthick=2.0,
            zorder=6
        )

    # Белый ореол под маркерами, чтобы верхние точки не терялись на 4096/3840.
    halo_size = 15 if is_main else 13

    ax.plot(
        xv,
        yv,
        marker='o',
        color='white',
        markersize=halo_size,
        lw=0,
        markeredgecolor='white',
        markeredgewidth=0,
        zorder=7 if not is_main else 10
    )

    # Основные маркеры
    m_face = color if is_main else 'white'
    m_edge = 'white' if is_main else color
    m_size = 10 if is_main else 8
    z_mark = 11 if is_main else 8

    ax.plot(
        xv,
        yv,
        marker='o',
        color=color,
        markersize=m_size,
        lw=0,
        markerfacecolor=m_face,
        markeredgecolor=m_edge,
        markeredgewidth=2.2,
        zorder=z_mark
    )

    for xi, yi, mos in zip(xv, yv, MOS_SCORES):
        if np.isnan(yi):
            continue

        if (fmt, mos) in CUSTOM_LABEL_POSITIONS:
            dx, dy = CUSTOM_LABEL_POSITIONS[(fmt, mos)]
            ha, va = 'center', 'center'
        else:
            # Верхнюю зону MOS=5 опускаем вниз, чтобы подписи не слипались с 4096/3840.
            if mos == 5:
                if fmt == 'Main':
                    dx, dy = -12, 20
                    ha, va = 'right', 'bottom'
                elif fmt == '720':
                    dx, dy = -8, -24
                    ha, va = 'right', 'top'
                elif fmt == '1080':
                    dx, dy = 8, -24
                    ha, va = 'left', 'top'
                elif fmt == '4K':
                    dx, dy = 12, -24
                    ha, va = 'left', 'top'
            else:
                if fmt == 'Main':
                    dx, dy = -12, 12
                    ha, va = 'right', 'bottom'
                elif fmt == '720':
                    dx, dy = -8, -14
                    ha, va = 'right', 'top'
                elif fmt == '1080':
                    dx, dy = 8, -14
                    ha, va = 'left', 'top'
                elif fmt == '4K':
                    dx, dy = 12, 12
                    ha, va = 'left', 'bottom'

        font_w = 'black' if is_main else 'bold'
        f_size = 9 if is_main else 8

        ax.annotate(
            f'{yi:.0f}',
            xy=(xi, yi),
            xytext=(dx, dy),
            textcoords='offset points',
            ha=ha,
            va=va,
            fontsize=f_size,
            color=color,
            fontweight=font_w,
            bbox=dict(
                boxstyle='square,pad=0.2',
                facecolor='white',
                edgecolor=color,
                lw=1.2 if is_main else 0.8,
                alpha=0.95
            ),
            zorder=12
        )

ax.set_xticks(MOS_SCORES)
ax.set_xticklabels([MOS_LABELS[m] for m in MOS_SCORES], fontsize=11, fontweight='bold')

minor_ticks = []

for m in MOS_SCORES:
    minor_ticks.extend([m - LANE_HALF_WIDTH, m + LANE_HALF_WIDTH])

ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
ax.tick_params(axis='x', which='minor', colors='#A0A0A0', length=8, width=1.2)
ax.tick_params(axis='x', which='major', length=0, pad=10)

apply_log_y_ticks(ax, Y_BOT, Y_TOP)

n_p = df['Испытуемый'].nunique()
n_s = df['Сессия'].nunique() if 'Сессия' in df.columns else '—'

ax.set_title(
    'Формирование базовой лестницы QoE: усреднение форматов экрана\n'
    'логарифмическая шкала разрешения',
    fontsize=15,
    fontweight='bold',
    color='#222222',
    pad=25
)

plt.figtext(
    0.5,
    0.90,
    f'Участников: {n_p}  |  Сессий: {n_s}  |  '
    'Сглаживание экспериментальных данных: PCHIP (монотонно кусочно-кубическая)  |  '
    'Вертикальные интервалы: 95% ДИ среднего по участникам',
    ha='center',
    fontsize=11,
    color='#555555'
)

ax.set_xlabel('Выделенные зоны субъективных оценок (MOS)', fontsize=12, labelpad=8)
ax.set_ylabel('Пороговое разрешение (ширина, пикс., лог. шкала)', fontsize=12, labelpad=12)

leg = ax.legend(
    fontsize=10.5,
    loc='lower left',
    bbox_to_anchor=(0.02, 0.02),
    framealpha=0.95,
    edgecolor='#BBBBBB',
    borderpad=0.8,
    labelspacing=0.6,
    handlelength=1.5
)

for line in leg.get_lines():
    line.set_linewidth(3.0)

plt.tight_layout()

plt.savefig(
    'qoe_average_ladder_logscale.png',
    dpi=200,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

plt.show()

print("\n✓ Логарифмический график формирования лестницы сохранён: qoe_average_ladder_logscale.png")
files.download('qoe_average_ladder_logscale.png')