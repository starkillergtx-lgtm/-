# -*- coding: utf-8 -*-
"""
График 3 — расширенная перцептивная QoE-лестница, 7 уровней.
Линейная шкала Y.

Новая компромиссная логика:
- график НЕ пересчитывает режимы 1080/720 как физические экраны;
- используется итоговая Main/QoE-кривая из графика 2;
- координата ширины — деградированное 4K-видео, приведённое из DCI 4096
  к UHD-шкале 3840×2160:
      width_norm = width_raw × 3840 / 4096
- линия 4096×2160 удалена;
- верхняя граница шкалы — 3840×2160.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from google.colab import files
except Exception:
    files = None

# ── 1. Исходные данные итоговой Main/QoE-кривой ────────────────────────
# Эти 4 точки соответствуют усреднённой зависимости из графика 2
# после нормировки к UHD-шкале 3840×2160.
base_mos_points = np.array([5.0, 4.0, 3.0, 2.0])
base_widths = np.array([
    3749.585097,  # MOS=5
     980.373171,  # MOS=4
     484.037966,  # MOS=3
     274.906834,  # MOS=2
])

# Полная 7-уровневая лестница: 4 экспериментальные MOS-точки + 3 субуровня.
mos_points = np.array([5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0])

# Высоты для подписей оставлены в той же логике, что и раньше:
# ширина приведена к UHD-шкале, высота соответствует исходной вертикали 2160.
# Основная метрика графика — ширина; высота используется только в подписи уровня.
height_labels = np.array([2109, 1100, 551, 322, 272, 263, 154])

DISPLAY_MAX_WIDTH = 3840
DISPLAY_MAX_HEIGHT = 2160

# Подписи MOS по шкале ITU-R BT.500.
mos_names = {
    5.0: '5.0\n(отлично)',
    4.5: '4.5\n(очень хорошо)',
    4.0: '4.0\n(хорошо)',
    3.5: '3.5\n(приемлемо)',
    3.0: '3.0\n(удовл.)',
    2.5: '2.5\n(неудовл.)',
    2.0: '2.0\n(плохо)',
}

# Эталонные уровни без 4096×2160: верхняя граница — UHD 3840×2160.
VIDEO_STANDARDS = [
    (3840, '3840×2160 (UHD 4K — верхняя граница шкалы)'),
    (2560, '2560×1440 (QHD / 2K)'),
    (1920, '1920×1080 (FHD)'),
    (1280, '1280×720 (HD 720p)'),
    (854,  '854×480 (SD 480p)'),
    (640,  '640×360 (360p)'),
    (426,  '426×240 (240p)'),
]

# ── 2. Интерполяционный полином Лагранжа ───────────────────────────────
def lagrange_eval(x_points, y_points, x_eval):
    """
    Явная интерполяция Лагранжа.
    Для четырёх опорных точек MOS=5, 4, 3, 2 получается единственный
    интерполяционный полином 3-й степени.

    ВАЖНО:
    - это не регрессия;
    - это не сплайн;
    - PCHIP не используется;
    - промежуточные субуровни MOS=4.5, 3.5, 2.5 рассчитаны по этому полиному.
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
    yd = np.clip(yd, 0, DISPLAY_MAX_WIDTH)
    return xd, yd


def lagrange_value_at(x_points, y_points, x_query):
    return float(np.clip(lagrange_eval(x_points, y_points, np.array([x_query]))[0], 0, DISPLAY_MAX_WIDTH))


# Полные значения 7 уровней рассчитываются от основной Main-кривой.
widths = lagrange_eval(base_mos_points, base_widths, mos_points)
widths = np.clip(widths, 0, DISPLAY_MAX_WIDTH)
labels = [f'{int(round(w))}×{int(h)}' for w, h in zip(widths, height_labels)]

# ── 3. Настройка графического поля ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 9.5), facecolor='white')
ax.set_facecolor('white')

ax.grid(True, which='major', color='#E0E0E0', lw=1.0, zorder=1)
ax.grid(True, which='minor', color='#F0F0F0', lw=0.6, zorder=1)
ax.minorticks_on()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(1.2)
    ax.spines[spine].set_color('#333333')

# Инверсия оси X: MOS=5 слева, MOS=2 справа.
X_R, X_L = 5.55, 0.8
Y_BOT, Y_TOP = 0, 4000
ax.set_xlim(X_R, X_L)
ax.set_ylim(Y_BOT, Y_TOP)

# ── 4. Справочные линии видеостандартов ────────────────────────────────
for res, name in VIDEO_STANDARDS:
    if Y_BOT <= res <= Y_TOP:
        ax.axhline(res, color='#A0A0A0', lw=1.2, ls='-.', alpha=0.85, zorder=2)
        text_y = res + 25
        if res == 3840:
            text_y = res - 40
            va = 'top'
        else:
            va = 'bottom'
        ax.text(
            0.85,
            text_y,
            name,
            ha='right',
            va=va,
            fontsize=9,
            color='#555555',
            style='italic',
            zorder=2,
        )

# ── 5. Построение мастер-кривой и маркеров ─────────────────────────────
xd, yd = lagrange_curve(base_mos_points, base_widths, num=500)

MASTER_COLOR = '#1A1A1A'
ax.plot(
    xd,
    yd,
    color=MASTER_COLOR,
    lw=3.5,
    label='Итоговая перцептивная QoE-кривая (полином Лагранжа)',
    zorder=4,
)

for xi, yi in zip(mos_points, widths):
    ax.hlines(yi, xi, X_R, colors=MASTER_COLOR, linestyles=':', lw=1.2, alpha=0.3, zorder=3)
    ax.vlines(xi, Y_BOT, yi, colors=MASTER_COLOR, linestyles=':', lw=1.2, alpha=0.3, zorder=3)

ax.plot(
    mos_points,
    widths,
    'o',
    color=MASTER_COLOR,
    markersize=11,
    markeredgecolor='white',
    markeredgewidth=2.5,
    zorder=6,
)

for xi, yi, lbl in zip(mos_points, widths, labels):
    if xi in (5.0, 2.5):
        y_offset = -28
        v_align = 'top'
    else:
        y_offset = 18
        v_align = 'bottom'

    ax.annotate(
        lbl,
        xy=(xi, yi),
        xytext=(0, y_offset),
        textcoords='offset points',
        ha='center',
        va=v_align,
        fontsize=9.5,
        color='#222222',
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor='#CCCCCC',
            lw=1.2,
            alpha=1.0,
        ),
        zorder=7,
    )

# ── 6. Оформление шкал и подписей ──────────────────────────────────────
ax.set_xticks(mos_points)
ax.set_xticklabels([mos_names[m] for m in mos_points], fontsize=9.0, fontweight='bold', color='#333333')

ax.yaxis.set_major_locator(mticker.MultipleLocator(250))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(50))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}' if v > 0 else '0'))

ax.set_title(
    'График 3: Расширенная перцептивная QoE-лестница (7 уровней)\n'
    'Координата: деградированное 4K-видео, приведённое к UHD-шкале 3840×2160\n'
    'Интерполяция: полином Лагранжа 3-й степени по 4 опорным точкам',
    fontsize=13,
    fontweight='bold',
    pad=22,
    color='#111111',
)
ax.set_xlabel('Субъективные оценки по шкале ITU-R BT.500 (MOS)', fontsize=11, labelpad=12, fontweight='bold', color='#222222')
ax.set_ylabel('Нормированное пороговое разрешение, ширина (пикс.)', fontsize=11, labelpad=12, fontweight='bold', color='#222222')

ax.legend(loc='upper right', fontsize=10, framealpha=1.0, edgecolor='#CCCCCC', facecolor='white')

print('\nДиагностика интерполяции Лагранжа для графика 3:')
print('  Опорные точки: MOS=5, 4, 3, 2')
print('  Координата: деградированное 4K-видео, приведённое к UHD-шкале 3840×2160')
print('  Субуровни рассчитаны по полиному, PCHIP / сплайн не используется')
print('  Линия 4096×2160 удалена; верхняя граница шкалы — 3840×2160')
for mos_q in [4.5, 3.5, 2.5]:
    w_q = lagrange_value_at(base_mos_points, base_widths, mos_q)
    print(f'  MOS={mos_q:.1f}: ширина ≈ {w_q:.1f} px, округление: {int(round(w_q))} px')

plt.tight_layout()

OUT_PNG = 'qoe_step3_7levels_lagrange_uhd.png'
plt.savefig(OUT_PNG, dpi=180, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

print(f'\n✓ График 3 успешно сгенерирован и сохранён как: {OUT_PNG}')

if files is not None:
    files.download(OUT_PNG)
