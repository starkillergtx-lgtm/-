import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'scipy', 'matplotlib', 'pandas', 'numpy'])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from google.colab import files
import io, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("Метод интерполяции: глобальный полином 3-й степени по 4 опорным точкам")
print("Форма вычисления: явная интерполяция Лагранжа")
print("PCHIP / сплайн: НЕ используется")

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
FORMATS = ['4K', '1080', '720']

COLORS = {
    '4K': '#1A6FBF',
    '1080': '#1E8A3A',
    '720': '#CC2020'
}

LINE_STYLES = {
    '4K': '-',
    '1080': '--',
    '720': '-.'
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
]

# ── 4. Расчёт статистики и ДИ ────────────────────────────────
# Среднее значение считается по очищенным наблюдениям.
# Доверительный интервал считается консервативно:
# сначала усредняем повторы внутри каждого участника,
# затем считаем 95% ДИ по участникам.

S = {}

FORMAT_COL = 'Formatted_screen_size' if 'Formatted_screen_size' in df.columns else 'Формат_экрана'
SUBJECT_COL = 'Испытуемый'

for fmt in FORMATS:
    S[fmt] = {}

    for mos in MOS_SCORES:
        mask = (df[FORMAT_COL] == fmt) & (df['MOS'] == mos)
        group = df.loc[mask, [SUBJECT_COL, 'width', 'height']].dropna()

        if len(group) == 0:
            mw, mh, ci, n_obs, n_subj = np.nan, np.nan, 0.0, 0, 0
        else:
            mw = group['width'].mean()
            mh = group['height'].mean()
            n_obs = len(group)

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

# ── 5. Интерполяционный полином Лагранжа ─────────────────────
def lagrange_eval(x_points, y_points, x_eval):
    """
    Явная интерполяция Лагранжа.
    Для четырёх опорных точек получается единственный полином 3-й степени.

    ВАЖНО:
    - это не регрессия;
    - это не сплайн;
    - PCHIP не используется;
    - визуальные горизонтальные сдвиги на сводном графике применяются
      только после расчёта полинома.
    """
    x_points = np.asarray(x_points, dtype=float)
    y_points = np.asarray(y_points, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    valid = ~(np.isnan(x_points) | np.isnan(y_points))
    x_points = x_points[valid]
    y_points = y_points[valid]

    order = np.argsort(x_points)
    x_points = x_points[order]
    y_points = y_points[order]

    y_eval = np.zeros_like(x_eval, dtype=float)

    for i in range(len(x_points)):
        basis = np.ones_like(x_eval, dtype=float)

        for j in range(len(x_points)):
            if i != j:
                basis *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])

        y_eval += y_points[i] * basis

    return y_eval

def lagrange_curve(x_points, y_points, num=500):
    x_points = np.asarray(x_points, dtype=float)
    y_points = np.asarray(y_points, dtype=float)

    valid = ~(np.isnan(x_points) | np.isnan(y_points))
    x_valid = x_points[valid]

    xd = np.linspace(np.min(x_valid), np.max(x_valid), num)
    yd = lagrange_eval(x_points, y_points, xd)

    return xd, yd

def lagrange_value_at(x_points, y_points, x_query):
    return float(lagrange_eval(x_points, y_points, np.array([x_query]))[0])

# ── 6. Оформление координатной сетки ─────────────────────────
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

# ── 7. Эталонные линии ITU ───────────────────────────────────
def draw_itu(ax, y_bot, y_top):
    x_pos = 1.17

    for res, label in ITU_LEVELS:
        if y_bot < res < y_top:
            ax.axhline(res, color='#BBBBBB', lw=1.1, ls='-.', zorder=1, alpha=0.75)
            ax.text(
                x_pos,
                res * 1.008,
                label,
                ha='right',
                va='bottom',
                fontsize=7.5,
                color='#777777',
                style='italic',
                zorder=3
            )

# ── 8. Функция построения одного формата ─────────────────────
def draw_single(ax, fmt, show_ylabel=True):
    color = COLORS[fmt]
    ls_style = LINE_STYLES[fmt]

    xv = np.array(MOS_SCORES, dtype=float)
    yv = np.array([S[fmt][m]['mean'] for m in MOS_SCORES])
    cv = np.array([S[fmt][m]['ci'] for m in MOS_SCORES])
    labels = [S[fmt][m]['label'] for m in MOS_SCORES]

    y_valid = yv[~np.isnan(yv)]
    Y_BOT = 0
    Y_TOP = int(np.ceil(max(y_valid) * 1.15 / 500.0) * 500)

    X_R, X_L = 5.55, 1.15

    ax.set_xlim(X_R, X_L)
    ax.set_ylim(Y_BOT, Y_TOP)

    apply_grid(ax)
    draw_itu(ax, Y_BOT, Y_TOP)

    # ── Реальные наблюдения ──
    mask_fmt = df[FORMAT_COL] == fmt

    scatter_x = []
    scatter_y = []

    for mos in MOS_SCORES:
        tmp = df[mask_fmt & (df['MOS'] == mos)]

        for val in tmp['width'].dropna():
            scatter_x.append(mos + np.random.uniform(-0.05, 0.05))
            scatter_y.append(val)

    ax.scatter(
        scatter_x,
        scatter_y,
        s=14,
        color='#808080',
        alpha=0.18,
        edgecolors='none',
        zorder=2
    )

    # Проекции к осям
    for xi, yi in zip(xv, yv):
        if np.isnan(yi):
            continue

        ax.hlines(
            yi,
            xi,
            X_R,
            colors=color,
            linestyles=':',
            lw=1.0,
            alpha=0.5,
            zorder=2
        )

        ax.vlines(
            xi,
            Y_BOT,
            yi,
            colors=color,
            linestyles=':',
            lw=1.0,
            alpha=0.5,
            zorder=2
        )

    # ── Интерполяционный полином Лагранжа ──
    xd, yd = lagrange_curve(xv, yv, num=500)

    ax.plot(
        xd,
        yd,
        color=color,
        ls=ls_style,
        lw=2.8,
        solid_capstyle='round',
        alpha=0.95,
        zorder=4
    )

    # Средние точки
    ax.plot(
        xv,
        yv,
        marker='o',
        color=color,
        markersize=11,
        lw=0,
        markeredgecolor='white',
        markeredgewidth=2.2,
        zorder=5
    )

    # Доверительные интервалы
    ax.errorbar(
        xv,
        yv,
        yerr=cv,
        fmt='none',
        ecolor=color,
        elinewidth=2.5,
        capsize=8,
        capthick=2.5,
        zorder=6
    )

    # Подписи средних значений
    for xi, yi, lbl in zip(xv, yv, labels):
        if np.isnan(yi):
            continue

        ax.annotate(
            lbl,
            xy=(xi, yi),
            xytext=(0, 12),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=9,
            color=color,
            fontweight='bold',
            bbox=dict(
                boxstyle='square,pad=0.25',
                facecolor='white',
                edgecolor=color,
                lw=1.2,
                alpha=0.95
            ),
            zorder=7
        )

    ax.set_xticks(MOS_SCORES)
    ax.set_xticklabels([MOS_LABELS[m].replace('\n', ' ') for m in MOS_SCORES], fontsize=10)

    ax.yaxis.set_major_locator(mticker.MultipleLocator(250))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(50))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}' if v > 0 else '0'))

    ax.set_title(f'Формат экрана: {fmt}', fontsize=13, fontweight='bold', color=color, pad=12)
    ax.set_xlabel('Субъективная оценка MOS (ITU-R BT.500)', fontsize=10, labelpad=8)

    if show_ylabel:
        ax.set_ylabel('Пороговое разрешение (ширина, пикс.)', fontsize=10, labelpad=8)

# ── 9. Функция построения сводного графика ───────────────────
def draw_combined(ax):
    y_all = [
        S[fmt][m]['mean']
        for fmt in FORMATS
        for m in MOS_SCORES
        if not np.isnan(S[fmt][m]['mean'])
    ]

    Y_BOT = 0
    Y_TOP = int(np.ceil(max(y_all) * 1.15 / 500.0) * 500)

    X_R, X_L = 5.55, 1.15

    ax.set_xlim(X_R, X_L)
    ax.set_ylim(Y_BOT, Y_TOP)

    apply_grid(ax)
    draw_itu(ax, Y_BOT, Y_TOP)

    LANE_HALF_WIDTH = 0.13

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

        ax.axvline(
            mos,
            color='#999999',
            lw=1.2,
            ls=':',
            zorder=1
        )

    # Сдвиги используются только для визуального разнесения кривых
    OFFSETS = {
        '720': 0.11,
        '1080': 0.0,
        '4K': -0.11
    }

    # Тонкие проекции к оси Y
    for fmt in FORMATS:
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

    # Отрисовка интерполяционных полиномов Лагранжа
    for fmt in FORMATS:
        color = COLORS[fmt]
        ls_style = LINE_STYLES[fmt]

        x_model = np.array(MOS_SCORES, dtype=float)
        yv = np.array([S[fmt][m]['mean'] for m in MOS_SCORES])
        cv = np.array([S[fmt][m]['ci'] for m in MOS_SCORES])

        # Полином строится по реальным MOS-точкам без визуального сдвига.
        # Сдвиг применяется только при отрисовке, чтобы разнести ряды.
        offset = OFFSETS[fmt]
        xv = x_model + offset
        xd_model, yd = lagrange_curve(x_model, yv, num=400)
        xd = xd_model + offset

        ax.plot(
            xd,
            yd,
            color=color,
            ls=ls_style,
            lw=2.5,
            alpha=0.95,
            solid_capstyle='round',
            zorder=3
        )

        ax.plot(
            xv,
            yv,
            marker='o',
            color=color,
            markersize=9,
            lw=0,
            markeredgecolor='white',
            markeredgewidth=2.0,
            zorder=5,
            label=fmt
        )

        ax.errorbar(
            xv,
            yv,
            yerr=cv,
            fmt='none',
            ecolor=color,
            elinewidth=2.2,
            capsize=6,
            capthick=2.2,
            zorder=6
        )

        for xi, yi, mos in zip(xv, yv, MOS_SCORES):
            if np.isnan(yi):
                continue

            if fmt == '720':
                dx, dy = -13, 14
                ha, va = 'right', 'bottom'
            elif fmt == '1080':
                dx, dy = 0, -18
                ha, va = 'center', 'top'
            else:
                dx, dy = 13, 14
                ha, va = 'left', 'bottom'

            ax.annotate(
                f'{yi:.0f}',
                xy=(xi, yi),
                xytext=(dx, dy),
                textcoords='offset points',
                ha=ha,
                va=va,
                fontsize=8,
                color=color,
                fontweight='bold',
                bbox=dict(
                    boxstyle='square,pad=0.15',
                    facecolor='white',
                    edgecolor=color,
                    lw=0.8,
                    alpha=0.95
                ),
                zorder=7
            )

    ax.set_xticks(MOS_SCORES)
    ax.set_xticklabels([MOS_LABELS[m] for m in MOS_SCORES], fontsize=10, fontweight='bold')

    minor_ticks = []
    for m in MOS_SCORES:
        minor_ticks.extend([m - LANE_HALF_WIDTH, m + LANE_HALF_WIDTH])

    ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
    ax.tick_params(axis='x', which='minor', colors='#A0A0A0', length=6, width=1.2)
    ax.tick_params(axis='x', which='major', length=0, pad=8)

    ax.yaxis.set_major_locator(mticker.MultipleLocator(250))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(50))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}' if v > 0 else '0'))

    ax.set_title('Сводный график — все форматы', fontsize=13, fontweight='bold', color='#222222', pad=12)
    ax.set_xlabel('Выделенные зоны субъективных оценок (MOS)', fontsize=10, labelpad=4)
    ax.set_ylabel('Пороговое разрешение (ширина, пикс.)', fontsize=10, labelpad=8)

    ax.legend(
        title='Формат экрана',
        fontsize=10,
        title_fontsize=10,
        loc='upper right',
        framealpha=0.95,
        edgecolor='#BBBBBB'
    )

# ── 10. Компоновка панели 2×2 ─────────────────────────────────
n_p = df['Испытуемый'].nunique()
n_s = df['Сессия'].nunique() if 'Сессия' in df.columns else '—'

print("\nДиагностика интерполяции Лагранжа:")
print("  Метод: глобальный полином 3-й степени по 4 опорным точкам")
print("  Форма вычисления: явная формула Лагранжа")
print("  PCHIP / сплайн: НЕ используется")
for fmt in FORMATS:
    x_check = np.array(MOS_SCORES, dtype=float)
    y_check = np.array([S[fmt][m]['mean'] for m in MOS_SCORES], dtype=float)

    print(f"\n{fmt}:")
    for mos_q in [4.5, 3.5, 2.5]:
        w_q = lagrange_value_at(x_check, y_check, mos_q)
        print(f"  MOS={mos_q:.1f}: ширина ≈ {w_q:.1f} px, округление: {int(round(w_q))} px")

fig, axes = plt.subplots(2, 2, figsize=(22, 16), facecolor='white')
fig.patch.set_facecolor('white')

fig.suptitle(
    'QoE-лестница: субъективная оценка качества видео vs пороговое разрешение\n'
    f'Участников: {n_p}  |  Сессий: {n_s}  |  '
    'Интерполяция: полином 3-й степени по 4 опорным точкам  |  '
    'Вертикальные интервалы: 95% ДИ среднего значения по участникам |  '
    'Штрихпунктир: стандартные уровни ITU-R (WxH)',
    fontsize=12,
    y=1.01,
    color='#222222'
)

draw_single(axes[0, 0], '4K', show_ylabel=True)
draw_single(axes[0, 1], '1080', show_ylabel=True)
draw_single(axes[1, 0], '720', show_ylabel=True)
draw_combined(axes[1, 1])

plt.tight_layout(h_pad=5, w_pad=4)

plt.savefig(
    'qoe_ultimate_lagrange_poly.png',
    dpi=180,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

plt.show()

print("\n✓ График сохранён: qoe_ultimate_lagrange_poly.png")
files.download('qoe_ultimate_lagrange_poly.png')