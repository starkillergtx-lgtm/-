import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from google.colab import files

# ── 1. Исходные данные: Полная 7-уровневая лестница ─────────────────────
mos_points = np.array([5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0])
widths     = np.array([4000, 2087, 1046, 610, 516, 499, 293])
heights    = np.array([2109, 1100, 551, 322, 272, 263, 154])
labels     = [f"{w}×{h}" for w, h in zip(widths, heights)]

# Опорные экспериментальные точки, по которым строится полином 3-й степени.
# Субуровни 4.5, 3.5 и 2.5 являются значениями этого полинома.
base_mos_points = np.array([5.0, 4.0, 3.0, 2.0])
base_widths     = np.array([4000, 1046, 516, 293])

# Академически выверенные подписи (основа: шкала ITU-R BT.500)
mos_names = {
    5.0: '5.0\n(отлично)',
    4.5: '4.5\n(очень хорошо)',
    4.0: '4.0\n(хорошо)',
    3.5: '3.5\n(приемлемо)',
    3.0: '3.0\n(удовл.)',
    2.5: '2.5\n(неудовл.)',
    2.0: '2.0\n(плохо)'
}

# Полные стандарты стриминга и видеоформатов
VIDEO_STANDARDS = [
    (4096, '4096×2160 (DCI 4K — исходное видео)'),
    (3840, '3840×2160 (UHD 4K — формат экрана)'),
    (2560, '2560×1440 (QHD / 2K)'),
    (1920, '1920×1080 (FHD)'),
    (1280, '1280×720 (HD 720p)'),
    (854,  '854×480 (SD 480p)'),
    (640,  '640×360 (360p)'),
    (426,  '426×240 (240p)'),
    (240,  '240×135 (нижняя граница лог. шкалы)')
]

LOG_TICKS = [240, 360, 480, 720, 1080, 1440, 2160, 3840, 4096, 5000]


# ── 2. Интерполяционный полином Лагранжа ─────────────────────
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

    return xd, yd

def lagrange_value_at(x_points, y_points, x_query):
    return float(lagrange_eval(x_points, y_points, np.array([x_query]))[0])

# ── 3. Настройка графического поля ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 9.5), facecolor='white')
ax.set_facecolor('white')

# Настройка сетки
ax.grid(True, which='major', color='#E0E0E0', lw=1.0, zorder=1)
ax.grid(True, which='minor', color='#F0F0F0', lw=0.6, zorder=1)
ax.minorticks_on()

# Очистка контуров
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(1.2)
    ax.spines[spine].set_color('#333333')

# Границы и логарифмическая шкала
X_R, X_L = 5.55, 0.8
Y_BOT, Y_TOP = 150, 5500

ax.set_xlim(X_R, X_L)
ax.set_ylim(Y_BOT, Y_TOP)
ax.set_yscale('log')

# ── 4. Справочные линии видеостандартов ─────────────────────────────────
for res, name in VIDEO_STANDARDS:
    if Y_BOT < res < Y_TOP:

        # Верхние стандарты 4096 и 3840 находятся близко,
        # поэтому линии делаем светлее и разводим подписи по вертикали.
        if res == 4096:
            line_alpha = 0.45
            text_y = res * 1.035
            text_va = 'bottom'
        elif res == 3840:
            line_alpha = 0.45
            text_y = res * 0.965
            text_va = 'top'
        elif res == 240:
            line_alpha = 0.55
            text_y = res * 0.93
            text_va = 'top'
        else:
            line_alpha = 0.85
            text_y = res * 1.015
            text_va = 'bottom'

        ax.axhline(
            res,
            color='#A0A0A0',
            lw=1.1,
            ls='-.',
            alpha=line_alpha,
            zorder=2
        )

        ax.text(
            0.85,
            text_y,
            name,
            ha='right',
            va=text_va,
            fontsize=8.2 if res == 240 else 9,
            color='#555555',
            style='italic',
            zorder=2
        )

# ── 5. Построение мастер-кривой и маркеров ──────────────────────────────
# Полином строится только по четырём экспериментальным опорным точкам MOS=5, 4, 3, 2.
# Промежуточные субуровни нанесены как значения этого полинома.
xd, yd = lagrange_curve(base_mos_points, base_widths, num=500)

MASTER_COLOR = '#1A1A1A'

valid_plot = yd > 0

ax.plot(
    xd[valid_plot],
    yd[valid_plot],
    color=MASTER_COLOR,
    lw=3.5,
    label='Финальная перцептивная QoE-кривая (полином Лагранжа)',
    zorder=4
)

# Тонкие пунктирные проекции к осям
for xi, yi in zip(mos_points, widths):
    ax.hlines(
        yi,
        xi,
        X_R,
        colors=MASTER_COLOR,
        linestyles=':',
        lw=1.2,
        alpha=0.3,
        zorder=3
    )

    ax.vlines(
        xi,
        Y_BOT,
        yi,
        colors=MASTER_COLOR,
        linestyles=':',
        lw=1.2,
        alpha=0.3,
        zorder=3
    )

# Белый ореол под маркерами
ax.plot(
    mos_points,
    widths,
    'o',
    color='white',
    markersize=16,
    markeredgecolor='white',
    markeredgewidth=0,
    zorder=5
)

# Основные маркеры
ax.plot(
    mos_points,
    widths,
    'o',
    color=MASTER_COLOR,
    markersize=11,
    markeredgecolor='white',
    markeredgewidth=2.5,
    zorder=6
)

# Контрастные информационные плашки
for xi, yi, lbl in zip(mos_points, widths, labels):

    # Верхнюю точку MOS=5 опускаем вниз, чтобы она не слипалась с 4096/3840.
    if xi == 5.0:
        y_offset = -28
        v_align = 'top'
    elif xi == 2.5:
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
            alpha=1.0
        ),
        zorder=7
    )

# ── 6. Оформление шкал и подписей ───────────────────────────────────────
ax.set_xticks(mos_points)
ax.set_xticklabels(
    [mos_names[m] for m in mos_points],
    fontsize=9.0,
    fontweight='bold',
    color='#333333'
)

ax.yaxis.set_major_locator(mticker.FixedLocator(LOG_TICKS))
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{int(v):,}' if Y_BOT <= v <= Y_TOP else '')
)

ax.yaxis.set_minor_locator(
    mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
)

# Заголовки
ax.set_title(
    'График 3: Расширенная перцептивная QoE-лестница (7 уровней)\n'
    'Сопоставление зрительных порогов с техническими стандартами потокового видео\nИнтерполяция: полином Лагранжа 3-й степени по 4 опорным точкам '
    '(логарифмическая шкала разрешения)',
    fontsize=13,
    fontweight='bold',
    pad=22,
    color='#111111'
)

ax.set_xlabel(
    'Субъективные оценки по шкале ITU-R BT.500 (MOS)',
    fontsize=11,
    labelpad=12,
    fontweight='bold',
    color='#222222'
)

ax.set_ylabel(
    'Пороговое разрешение (ширина, пикс., лог. шкала)',
    fontsize=11,
    labelpad=12,
    fontweight='bold',
    color='#222222'
)

# Легенда
ax.legend(
    loc='upper right',
    fontsize=10,
    framealpha=1.0,
    edgecolor='#CCCCCC',
    facecolor='white'
)


print("\nДиагностика интерполяции Лагранжа для графика 3:")
print("  Опорные точки: MOS=5, 4, 3, 2")
print("  Субуровни рассчитаны по полиному, PCHIP / сплайн не используется")
for mos_q in [4.5, 3.5, 2.5]:
    w_q = lagrange_value_at(base_mos_points, base_widths, mos_q)
    print(f"  MOS={mos_q:.1f}: ширина ≈ {w_q:.1f} px, округление: {int(round(w_q))} px")

# ── 7. Вывод и сохранение ───────────────────────────────────────────────
plt.tight_layout()

plt.savefig(
    'qoe_step3_7levels_lagrange_poly_log.png',
    dpi=180,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

plt.show()

print("\n✓ Логарифмический график 3 успешно сгенерирован и сохранён как: qoe_step3_7levels_lagrange_poly_log.png")
files.download('qoe_step3_7levels_lagrange_poly_log.png')