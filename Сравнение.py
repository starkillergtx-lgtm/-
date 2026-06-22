# -*- coding: utf-8 -*-
"""
График сравнения — предлагаемая перцептивная лестница vs YouTube / Netflix.

Новая компромиссная логика:
- авторская зависимость разрешение(MOS) строится по итоговой Main/QoE-кривой,
  приведённой из DCI 4096 к UHD-шкале 3840×2160;
- НЕ пересчитываем режимы 1080/720 как физические экраны;
- внешние лестницы YouTube и Netflix остаются техническими дискретными уровнями;
- каждый внешний уровень переносится на авторскую MOS-зависимость по ширине;
- линия 4096×2160 удалена;
- уровень YouTube 3840×2160 немного выше верхней экспериментальной точки
  авторской шкалы, поэтому на графике ограничивается MOS=5.0 без экстраполяции.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import brentq

try:
    from google.colab import files
except Exception:
    files = None

print("Авторская зависимость разрешение(MOS): полином Лагранжа 3-й степени по 4 опорным точкам")
print("Координата авторской шкалы: деградированное 4K-видео, приведённое к UHD 3840×2160")
print("YouTube и Netflix: внешние дискретные лестницы; их уровни переносятся на авторскую MOS-зависимость")
print("PCHIP / сплайн: НЕ используется")

DISPLAY_MAX_WIDTH = 3840
DISPLAY_MAX_HEIGHT = 2160

# ─────────────────────────────────────────────────────────────────────
# 1. Итоговая экспериментальная зависимость разрешение(MOS)
# ─────────────────────────────────────────────────────────────────────
# Четыре экспериментальные опорные точки Main/QoE-кривой после нормировки к UHD.
# По ним строится единственный полином Лагранжа 3-й степени.
mos_base = np.array([5.0, 4.0, 3.0, 2.0])
width_base = np.array([
    3749.585097,  # MOS=5
     980.373171,  # MOS=4
     484.037966,  # MOS=3
     274.906834,  # MOS=2
])

# Итоговая 6-уровневая перцептивная лестница.
# MOS=4.5 и MOS=3.5 рассчитываются по авторскому полиному.
# MOS=2.5 исключён из итоговой лестницы.
our_mos = np.array([5.0, 4.5, 4.0, 3.5, 3.0, 2.0])
our_heights = np.array([2109, 1100, 551, 322, 272, 154])

# ─────────────────────────────────────────────────────────────────────
# 2. Интерполяционный полином Лагранжа только для авторской зависимости
# ─────────────────────────────────────────────────────────────────────
def lagrange_eval(x_points, y_points, x_eval):
    """
    Явная интерполяция Лагранжа.
    Для четырёх опорных точек MOS=5, 4, 3, 2 получается единственный
    полином 3-й степени.

    ВАЖНО:
    - это используется только для авторской зависимости разрешение(MOS);
    - это не регрессия;
    - это не сплайн;
    - PCHIP не используется.
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


def perceptual_curve(num=900):
    xd = np.linspace(2.0, 5.0, num)
    yd = lagrange_eval(mos_base, width_base, xd)
    yd = np.clip(yd, 0, DISPLAY_MAX_WIDTH)
    return xd, yd


def perceptual_width_at_mos(mos_value):
    return float(np.clip(lagrange_eval(mos_base, width_base, np.array([mos_value]))[0], 0, DISPLAY_MAX_WIDTH))


def width_to_mos(width):
    """
    Перенос внешнего технического уровня на авторскую зависимость.

    Для ширин внутри диапазона авторской кривой решается обратная задача:
    width = Rпор(MOS).

    Для 3840×2160 YouTube ширина чуть выше верхней экспериментальной точки
    авторской шкалы (~3750 px), поэтому значение не экстраполируется за MOS=5,
    а ограничивается MOS=5.0.
    """
    y_min = float(np.min(width_base))
    y_max = float(np.max(width_base))

    if width > y_max:
        return 5.0, "выше верхней экспериментальной точки; на графике ограничено MOS=5.0"
    if width < y_min:
        return np.nan, "ниже диапазона авторской зависимости"

    mos = brentq(
        lambda x: float(lagrange_eval(mos_base, width_base, np.array([x]))[0]) - width,
        2.0,
        5.0,
    )
    return float(mos), "внутри диапазона авторской зависимости"


x_curve, y_curve = perceptual_curve(num=900)

our_widths = np.array([perceptual_width_at_mos(m) for m in our_mos])
our_graph_labels = [f"{int(round(w))}×{int(h)}" for w, h in zip(our_widths, our_heights)]
our_full_names = [
    f"MOS {m:.1f} - {label} - {res}"
    for m, label, res in zip(
        our_mos,
        ["отлично", "очень хорошо", "хорошо", "приемлемо", "удовлетворительно", "плохо"],
        our_graph_labels,
    )
]

# Диагностика субуровней.
print("\nДиагностика авторской перцептивной лестницы:")
for mos_q in [4.5, 3.5, 2.5]:
    w_q = perceptual_width_at_mos(mos_q)
    status = "включён" if mos_q in [4.5, 3.5] else "исключён из итоговой лестницы"
    print(f"  MOS={mos_q:.1f}: ширина ≈ {w_q:.1f} px, округление: {int(round(w_q))} px - {status}")

# ─────────────────────────────────────────────────────────────────────
# 3. Уровни YouTube и Netflix
# ─────────────────────────────────────────────────────────────────────
YOUTUBE_LEVELS = [
    {"graph_label": "3840×2160", "full_name": "2160p / UHD 4K / 3840×2160", "width": 3840, "height": 2160, "note": "стандартный уровень YouTube 16:9"},
    {"graph_label": "2560×1440", "full_name": "1440p / QHD / 2560×1440", "width": 2560, "height": 1440, "note": "стандартный уровень YouTube 16:9"},
    {"graph_label": "1920×1080", "full_name": "1080p / FHD / 1920×1080", "width": 1920, "height": 1080, "note": "стандартный уровень YouTube 16:9"},
    {"graph_label": "1280×720", "full_name": "720p / HD / 1280×720", "width": 1280, "height": 720, "note": "стандартный уровень YouTube 16:9"},
    {"graph_label": "854×480", "full_name": "480p / SD 480p / 854×480", "width": 854, "height": 480, "note": "стандартный уровень YouTube 16:9"},
    {"graph_label": "640×360", "full_name": "360p / SD 360p / 640×360", "width": 640, "height": 360, "note": "стандартный уровень YouTube 16:9"},
    {"graph_label": "426×240", "full_name": "240p / SD 240p / 426×240", "width": 426, "height": 240, "note": "стандартный уровень YouTube 16:9"},
]

NETFLIX_LEVELS = [
    {"graph_label": "1920×1080", "full_name": "1080p / FHD / 1920×1080", "width": 1920, "height": 1080, "note": "кандидатное разрешение Netflix per-title"},
    {"graph_label": "1280×720", "full_name": "720p / HD / 1280×720", "width": 1280, "height": 720, "note": "кандидатное разрешение Netflix per-title"},
    {"graph_label": "720×480", "full_name": "480p / NTSC SD / 720×480", "width": 720, "height": 480, "note": "кандидатное разрешение Netflix per-title"},
    {"graph_label": "512×384", "full_name": "384p / 4:3 / 512×384", "width": 512, "height": 384, "note": "кандидатное разрешение Netflix per-title"},
    {"graph_label": "384×288", "full_name": "288p / 4:3 / 384×288", "width": 384, "height": 288, "note": "кандидатное разрешение Netflix per-title"},
    {"graph_label": "320×240", "full_name": "240p / QVGA / 320×240", "width": 320, "height": 240, "note": "кандидатное разрешение Netflix per-title"},
]

# ─────────────────────────────────────────────────────────────────────
# 4. Таблица сопоставления
# ─────────────────────────────────────────────────────────────────────
def make_ladder_df(ladder_name, levels):
    rows = []
    for item in levels:
        mos, status = width_to_mos(item["width"])
        rows.append({
            "Лестница": ladder_name,
            "Уровень": item["full_name"],
            "Ширина_px": item["width"],
            "Высота_px": item["height"],
            "MOS_по_авторской_зависимости": mos,
            "Статус_переноса": status,
            "Комментарий": item["note"],
        })

    df = pd.DataFrame(rows)
    df["ΔMOS_к_следующему"] = df["MOS_по_авторской_зависимости"].diff(-1).abs()
    return df


df_our = pd.DataFrame({
    "Лестница": "Предлагаемая",
    "Уровень": our_full_names,
    "Ширина_px": np.round(our_widths).astype(int),
    "Высота_px": our_heights,
    "MOS_по_авторской_зависимости": our_mos,
    "Статус_переноса": ["уровень итоговой перцептивной лестницы"] * len(our_mos),
    "Комментарий": ["уровень итоговой перцептивной лестницы"] * len(our_mos),
})
df_our["ΔMOS_к_следующему"] = df_our["MOS_по_авторской_зависимости"].diff(-1).abs()

df_youtube = make_ladder_df("YouTube", YOUTUBE_LEVELS)
df_netflix = make_ladder_df("Netflix per-title candidates", NETFLIX_LEVELS)

df_all = pd.concat([df_our, df_youtube, df_netflix], ignore_index=True)
df_all["MOS_по_авторской_зависимости"] = df_all["MOS_по_авторской_зависимости"].round(3)
df_all["ΔMOS_к_следующему"] = df_all["ΔMOS_к_следующему"].round(3)

OUT_CSV = "comparison_ladders_mos_lagrange_uhd.csv"
df_all.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("\nСопоставление уровней по авторской зависимости разрешение(MOS):")
print(df_all.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
# 5. Оформление
# ─────────────────────────────────────────────────────────────────────
COLORS = {
    "ours": "#111111",
    "youtube": "#1A6FBF",
    "netflix": "#CC2020",
    "reference": "#A8A8A8",
}

LINE_STYLES = {
    "ours": "-",
    "youtube": "--",
    "netflix": "-.",
}

REFERENCE_RESOLUTION_LINES = [
    (3840, "3840×2160  (UHD 4K — верхняя граница шкалы)"),
    (2560, "2560×1440  (QHD / 2K)"),
    (1920, "1920×1080  (FHD)"),
    (1280, "1280×720   (HD 720p)"),
    (854,  " 854×480   (SD 480p)"),
    (640,  " 640×360   (360p)"),
    (426,  " 426×240   (240p)"),
    (320,  " 320×240   (QVGA)"),
]

MOS_ZONE_CENTERS = [5.0, 4.5, 4.0, 3.5, 3.0, 2.0]
LANE_HALF_WIDTH = 0.145

# Сдвиги только визуальные.
OFFSETS = {
    "ours": 0.00,
    "youtube": -0.090,
    "netflix": 0.090,
}

# ── Ручная раскладка плашек ───────────────────────────────────────────
OUR_LABEL_POS = {
    5.0: (-14, -30, "right", "top"),
    4.5: (28, 24, "left", "bottom"),
    4.0: (-24, 22, "right", "bottom"),
    3.5: (32, 22, "left", "bottom"),
    3.0: (-28, 28, "right", "bottom"),
    2.0: (-34, 26, "right", "bottom"),
}

YOUTUBE_LABEL_POS = {
    3840: (-16, 24, "right", "bottom"),
    2560: (-26, -26, "right", "top"),
    1920: (-28, -28, "right", "top"),
    1280: (-30, -30, "right", "top"),
    854:  (-34, 24, "right", "bottom"),
    640:  (-34, -30, "right", "top"),
    426:  (24, 26, "left", "bottom"),
}

NETFLIX_LABEL_POS = {
    1920: (26, 20, "left", "bottom"),
    1280: (28, 24, "left", "bottom"),
    720:  (30, -28, "left", "top"),
    512:  (30, 34, "left", "bottom"),
    384:  (36, 20, "left", "bottom"),
    320:  (-30, -22, "right", "top"),
}

fig, ax = plt.subplots(figsize=(19, 10.8), facecolor="white")
fig.patch.set_facecolor("white")
ax.set_facecolor("#FDFDFD")

ax.grid(True, which="major", color="#E2E2E2", lw=0.9, ls="-", zorder=0)
ax.grid(True, which="minor", color="#EEEEEE", lw=0.5, ls="-", zorder=0)
ax.minorticks_on()
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for sp in ["left", "bottom"]:
    ax.spines[sp].set_linewidth(1.2)
    ax.spines[sp].set_color("#444444")

X_R, X_L = 5.43, 0.78
Y_BOT, Y_TOP = 0, 4100

ax.set_xlim(X_R, X_L)
ax.set_ylim(Y_BOT, Y_TOP)

# ─────────────────────────────────────────────────────────────────────
# 6. MOS-зоны
# ─────────────────────────────────────────────────────────────────────
for mos in MOS_ZONE_CENTERS:
    ax.axvspan(
        mos - LANE_HALF_WIDTH,
        mos + LANE_HALF_WIDTH,
        facecolor="#F0F0F0",
        edgecolor="#D5D5D5",
        lw=1.0,
        alpha=0.72,
        zorder=1,
    )
    ax.axvline(mos, color="#999999", lw=1.1, ls=":", alpha=0.8, zorder=1)

# ─────────────────────────────────────────────────────────────────────
# 7. Справочные горизонтальные уровни разрешения
# ─────────────────────────────────────────────────────────────────────
for res, label in REFERENCE_RESOLUTION_LINES:
    if Y_BOT < res < Y_TOP:
        ax.axhline(res, color=COLORS["reference"], lw=1.1, ls="-.", zorder=1, alpha=0.72)
        ax.text(
            0.82,
            res + 22,
            label,
            ha="right",
            va="bottom",
            fontsize=8.2,
            color="#777777",
            style="italic",
            zorder=2,
        )

# ─────────────────────────────────────────────────────────────────────
# 8. Авторская зависимость и итоговые уровни
# ─────────────────────────────────────────────────────────────────────
ax.plot(
    x_curve,
    y_curve,
    color=COLORS["ours"],
    lw=3.5,
    ls=LINE_STYLES["ours"],
    zorder=5,
)

our_x_plot = our_mos + OFFSETS["ours"]

ax.plot(
    our_x_plot,
    our_widths,
    marker="o",
    markersize=10,
    lw=0,
    color=COLORS["ours"],
    markerfacecolor=COLORS["ours"],
    markeredgecolor="white",
    markeredgewidth=2.1,
    zorder=9,
)

for x, y in zip(our_x_plot, our_widths):
    ax.hlines(y, x, X_R, colors=COLORS["ours"], linestyles=":", lw=1.0, alpha=0.18, zorder=2)
    ax.vlines(x, Y_BOT, y, colors=COLORS["ours"], linestyles=":", lw=1.0, alpha=0.18, zorder=2)

for x, y, lbl, mos in zip(our_x_plot, our_widths, our_graph_labels, our_mos):
    dx, dy, ha, va = OUR_LABEL_POS[float(mos)]
    ax.annotate(
        lbl,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=7.6,
        color=COLORS["ours"],
        fontweight="bold",
        bbox=dict(
            boxstyle="square,pad=0.20",
            facecolor="white",
            edgecolor=COLORS["ours"],
            lw=1.0,
            alpha=0.96,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color=COLORS["ours"],
            lw=0.45,
            alpha=0.18,
            shrinkA=2,
            shrinkB=2,
        ),
        zorder=11,
    )

# ─────────────────────────────────────────────────────────────────────
# 9. Функция отрисовки внешних лестниц
# ─────────────────────────────────────────────────────────────────────
def plot_external_ladder(levels, color, marker, line_style, offset, label_positions):
    """
    YouTube и Netflix здесь не аппроксимируются полиномом.
    Они имеют собственный набор дискретных уровней. Каждый уровень переносится
    на ось MOS через авторскую зависимость, после чего точки соединяются ломаной
    линией только для визуального сопоставления порядка уровней.
    """
    true_mos = []
    statuses = []
    for item in levels:
        mos, status = width_to_mos(item["width"])
        true_mos.append(mos)
        statuses.append(status)

    true_mos = np.array(true_mos, dtype=float)
    widths = np.array([item["width"] for item in levels])
    labels = [item["graph_label"] for item in levels]

    plot_mos = true_mos + offset

    valid = ~np.isnan(plot_mos)

    # Ломаная по дискретным уровням внешней лестницы, не сглаженная интерполяция.
    ax.plot(
        plot_mos[valid],
        widths[valid],
        color=color,
        lw=2.10,
        ls=line_style,
        marker=marker,
        markersize=7.8,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.9,
        alpha=0.96,
        zorder=7,
    )

    for x, y in zip(plot_mos, widths):
        if np.isnan(x):
            continue
        ax.hlines(y, x, X_R, colors=color, linestyles=":", lw=0.9, alpha=0.16, zorder=2)
        ax.vlines(x, Y_BOT, y, colors=color, linestyles=":", lw=0.9, alpha=0.16, zorder=2)

    for x, y, txt in zip(plot_mos, widths, labels):
        if np.isnan(x):
            continue

        dx, dy, ha, va = label_positions.get(int(y), (10, 12, "left", "bottom"))
        ax.annotate(
            txt,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=6.95,
            color=color,
            fontweight="bold",
            bbox=dict(
                boxstyle="square,pad=0.13",
                facecolor="white",
                edgecolor=color,
                lw=0.75,
                alpha=0.96,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.45,
                alpha=0.18,
                shrinkA=2,
                shrinkB=2,
            ),
            zorder=10,
        )

    return true_mos, widths, statuses


plot_external_ladder(
    YOUTUBE_LEVELS,
    color=COLORS["youtube"],
    marker="s",
    line_style=LINE_STYLES["youtube"],
    offset=OFFSETS["youtube"],
    label_positions=YOUTUBE_LABEL_POS,
)

plot_external_ladder(
    NETFLIX_LEVELS,
    color=COLORS["netflix"],
    marker="D",
    line_style=LINE_STYLES["netflix"],
    offset=OFFSETS["netflix"],
    label_positions=NETFLIX_LABEL_POS,
)

# ─────────────────────────────────────────────────────────────────────
# 10. Шкалы, подписи, легенда
# ─────────────────────────────────────────────────────────────────────
ax.set_xticks(MOS_ZONE_CENTERS)
ax.set_xticklabels(
    [
        "5.0\n(отлично)",
        "4.5\n(очень хорошо)",
        "4.0\n(хорошо)",
        "3.5\n(приемлемо)",
        "3.0\n(удовл.)",
        "2.0\n(плохо)",
    ],
    fontsize=9.5,
    fontweight="bold",
)

minor_ticks = []
for m in MOS_ZONE_CENTERS:
    minor_ticks.extend([m - LANE_HALF_WIDTH, m + LANE_HALF_WIDTH])

ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_ticks))
ax.tick_params(axis="x", which="minor", colors="#A0A0A0", length=6, width=1.1)
ax.tick_params(axis="x", which="major", length=0, pad=9)

ax.yaxis.set_major_locator(mticker.MultipleLocator(500))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(100))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v > 0 else "0"))

ax.set_title(
    "Сопоставление перцептивной лестницы с техническими уровнями YouTube и Netflix\n"
    "Уровни внешних лестниц перенесены на экспериментальную зависимость разрешение(MOS)",
    fontsize=14,
    fontweight="bold",
    pad=22,
    color="#222222",
)

plt.figtext(
    0.43,
    0.925,
    "Авторская кривая: полином Лагранжа по 4 MOS-порогам  |  "
    "координата: деградированное 4K-видео, приведённое к UHD 3840×2160  |  "
    "YouTube и Netflix: дискретные уровни, соединённые ломаной",
    ha="center",
    fontsize=10.5,
    color="#555555",
)

ax.set_xlabel("Субъективная оценка MOS по построенной зависимости", fontsize=12, labelpad=12)
ax.set_ylabel("Нормированное пороговое разрешение, ширина (пикс.)", fontsize=12, labelpad=12)

legend_handles = [
    Line2D([0], [0], color=COLORS["ours"], lw=3.2, label="Предлагаемая зависимость Rпор(MOS)"),
    Line2D([0], [0], color=COLORS["ours"], marker="o", lw=0, markersize=8, markerfacecolor=COLORS["ours"], markeredgecolor="white", label="Итоговая перцептивная лестница"),
    Line2D([0], [0], color=COLORS["youtube"], lw=2.10, ls=LINE_STYLES["youtube"], marker="s", markersize=7, markerfacecolor="white", markeredgecolor=COLORS["youtube"], label="YouTube: дискретные уровни 16:9"),
    Line2D([0], [0], color=COLORS["netflix"], lw=2.10, ls=LINE_STYLES["netflix"], marker="D", markersize=7, markerfacecolor="white", markeredgecolor=COLORS["netflix"], label="Netflix: дискретные кандидаты per-title"),
    Line2D([0], [0], color=COLORS["reference"], lw=1.2, ls="-.", label="Справочные стандартные разрешения"),
    Patch(facecolor="#F0F0F0", edgecolor="#D5D5D5", label="Серые вертикальные зоны MOS"),
]

leg = ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(1.005, 1.0),
    fontsize=9.8,
    framealpha=0.96,
    edgecolor="#BBBBBB",
    facecolor="white",
    borderpad=0.8,
    labelspacing=0.65,
    handlelength=2.3,
)

for line in leg.get_lines():
    line.set_linewidth(2.4)

plt.tight_layout(rect=[0, 0, 0.80, 0.90])

OUT_PNG = "qoe_compare_youtube_netflix_lagrange_uhd.png"
plt.savefig(
    OUT_PNG,
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)

plt.show()

print(f"\n✓ График сохранён: {OUT_PNG}")
print(f"✓ Таблица сохранена: {OUT_CSV}")

if files is not None:
    files.download(OUT_PNG)
    files.download(OUT_CSV)
