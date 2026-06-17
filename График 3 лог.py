import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import PchipInterpolator
from google.colab import files

# ── 1. Исходные данные: Полная 7-уровневая лестница ─────────────────────
mos_points = np.array([5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0])
widths     = np.array([4000, 2114, 1046, 708, 516, 374, 293])
heights    = np.array([2109, 1115, 551, 373, 272, 197, 154])
labels     = [f"{w}×{h}" for w, h in zip(widths, heights)]

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
    (240,  '240 px (нижняя граница лог. шкалы)')
]

LOG_TICKS = [240, 360, 480, 720, 1080, 1440, 2160, 3840, 4096, 5000]

# ── 2. Настройка графического поля ─────────────────────────────────────
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

# ── 3. Справочные линии видеостандартов ─────────────────────────────────
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
            fontsize=9,
            color='#555555',
            style='italic',
            zorder=2
        )

# ── 4. Построение мастер-кривой и маркеров ──────────────────────────────
sort_idx = np.argsort(mos_points)
cs = PchipInterpolator(mos_points[sort_idx], widths[sort_idx])
xd = np.linspace(2.0, 5.0, 500)

MASTER_COLOR = '#1A1A1A'

ax.plot(
    xd,
    cs(xd),
    color=MASTER_COLOR,
    lw=3.5,
    label='Финальная перцептивная QoE-кривая (усредненная)',
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

# ── 5. Оформление шкал и подписей ───────────────────────────────────────
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
    'Сопоставление зрительных порогов с техническими стандартами потокового видео '
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

# ── 6. Вывод и сохранение ───────────────────────────────────────────────
plt.tight_layout()

plt.savefig(
    'qoe_step3_7levels_streaming_standards_logscale.png',
    dpi=180,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

plt.show()

print("\n✓ Логарифмический график 3 успешно сгенерирован и сохранён как: qoe_step3_7levels_streaming_standards_logscale.png")
files.download('qoe_step3_7levels_streaming_standards_logscale.png')