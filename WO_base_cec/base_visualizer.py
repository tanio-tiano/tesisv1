from pathlib import Path
import json
import re

import pandas as pd


def create_base_monitor(
    values_dir,
    graphs_dir,
    curve_glob="conv_curve_F*.csv",
    monitor_filename="interactive_monitor_base_all_functions.html",
    page_title="Monitor interactivo WO Base",
    chart_title="Curva de convergencia WO base",
    hint_text="Curva generada desde los CSV de `WO_base_cec/all_functions_outputs_final/values`.",
):
    values_dir = Path(values_dir)
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    functions = {}
    for curve_path in sorted(values_dir.glob(curve_glob)):
        match = re.search(r"F(\d+)", curve_path.stem)
        if match is None:
            continue
        function_id = int(match.group(1))
        curve_df = pd.read_csv(curve_path)
        if curve_df.empty:
            continue
        convergence = [float(v) for v in curve_df["best_fitness"].tolist()]
        functions[f"F{function_id}"] = {
            "function_id": function_id,
            "convergence": convergence,
            "initial_fitness": convergence[0],
            "final_fitness": convergence[-1],
            "iterations": len(convergence),
        }

    if not functions:
        return None

    ordered_functions = dict(
        sorted(functions.items(), key=lambda item: item[1]["function_id"])
    )

    monitor_path = graphs_dir / monitor_filename
    monitor_path.write_text(
        _build_html(
            {"functions": ordered_functions},
            page_title=page_title,
            chart_title=chart_title,
            hint_text=hint_text,
        ),
        encoding="utf-8",
    )
    return monitor_path


def _build_html(payload, page_title, chart_title, hint_text):
    html = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__PAGE_TITLE__</title>
  <style>
    :root { --panel:#fffaf0; --ink:#1f2937; --muted:#6b7280; --grid:#d7d2c8; --line:#1d4ed8; --accent:#111827; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Segoe UI", Tahoma, sans-serif; background: linear-gradient(180deg, #f7f3ea 0%, #efe8d8 100%); color: var(--ink); }
    .wrap { max-width: 1300px; margin: 0 auto; padding: 20px; }
    .topbar { display: grid; grid-template-columns: 1fr auto auto auto; gap: 12px; align-items: center; margin-bottom: 18px; }
    .title { font-size: 26px; font-weight: 700; }
    .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    button, select { border: 0; background: var(--accent); color: white; padding: 10px 14px; cursor: pointer; font-size: 14px; border-radius: 10px; }
    input[type="range"] { width: 360px; }
    .badge { background: rgba(17,24,39,0.08); border-radius: 999px; padding: 8px 12px; font-size: 13px; color: var(--muted); }
    .meta { display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 10px; margin-bottom: 12px; }
    .panel { background: var(--panel); border: 1px solid rgba(0,0,0,0.06); border-radius: 18px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.06); }
    .panel h2 { margin: 0 0 8px 0; font-size: 18px; }
    .meta-card { background: rgba(255,255,255,0.7); border: 1px solid rgba(0,0,0,0.05); border-radius: 12px; padding: 10px 12px; }
    .meta-card .k { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .meta-card .v { font-size: 16px; font-weight: 700; }
    svg { width: 100%; height: auto; display: block; background: white; border-radius: 14px; border: 1px solid rgba(0,0,0,0.05); }
    .hint { margin-top: 10px; font-size: 13px; color: var(--muted); }
    @media (max-width: 900px) { .topbar { grid-template-columns: 1fr; } input[type="range"] { width: 100%; } .meta { grid-template-columns: repeat(2, minmax(0,1fr)); } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="title">__PAGE_TITLE__</div>
      <div class="controls"><select id="functionSelect"></select></div>
      <div class="controls"><button id="prevBtn">Anterior</button><button id="nextBtn">Siguiente</button></div>
      <div class="controls"><input id="iterSlider" type="range" min="0" max="0" value="0"></div>
    </div>
    <div class="meta">
      <div class="meta-card"><div class="k">Funcion</div><div class="v" id="functionValue">-</div></div>
      <div class="meta-card"><div class="k">Iteracion</div><div class="v" id="iterationValue">-</div></div>
      <div class="meta-card"><div class="k">Fitness actual</div><div class="v" id="fitnessValue">-</div></div>
      <div class="meta-card"><div class="k">Fitness final</div><div class="v" id="finalValue">-</div></div>
      <div class="meta-card"><div class="k">Escala grafico</div><div class="v" id="scaleValue">-</div></div>
    </div>
    <div class="panel">
      <h2>__CHART_TITLE__</h2>
      <svg id="convSvg" viewBox="0 0 1000 460" preserveAspectRatio="none"></svg>
      <div class="hint">__HINT_TEXT__</div>
    </div>
  </div>
  <script>
    const ALL_DATA = __PAYLOAD__;
    let selectedFunction = null, DATA = null, maxIteration = 0;
    const $ = id => document.getElementById(id);
    const functionSelect = $("functionSelect"), slider = $("iterSlider"), convSvg = $("convSvg");
    function fmt(v) {
      const n = Number(v);
      if (!Number.isFinite(n)) return "inf";
      if (Math.abs(n) >= 1e4 || (Math.abs(n) > 0 && Math.abs(n) < 1e-3)) return n.toExponential(3);
      return n.toFixed(6);
    }
    function svgEl(name, attrs = {}, text = null) { const el = document.createElementNS("http://www.w3.org/2000/svg", name); for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v); if (text !== null) el.textContent = text; return el; }
    function buildScale(values) {
      const finiteValues = values.filter(v => Number.isFinite(v));
      const yMin = Math.min(...finiteValues), yMax = Math.max(...finiteValues);
      const safeMin = Math.max(Math.min(...finiteValues.filter(v => v > 0)), 1e-12);
      const positiveOnly = finiteValues.every(v => v > 0);
      const ratio = positiveOnly ? (yMax / safeMin) : 1;
      const useLog = positiveOnly && ratio >= 1e3;
      if (useLog) {
        const logMin = Math.log10(safeMin);
        const logMax = Math.log10(Math.max(yMax, safeMin * 1.000001));
        return {
          mode: "log10",
          label: "log10",
          yMap: v => Math.log10(Math.max(v, safeMin)),
          tickValues: Array.from({ length: 6 }, (_, i) => Math.pow(10, logMax - (i / 5) * (logMax - logMin))),
        };
      }
      const yRange = Math.max(yMax - yMin, 1e-9);
      return {
        mode: "linear",
        label: "lineal",
        yMap: v => v,
        tickValues: Array.from({ length: 6 }, (_, i) => yMax - (i / 5) * yRange),
      };
    }
    function setFunction(name) {
      selectedFunction = name;
      DATA = ALL_DATA.functions[name];
      maxIteration = DATA.convergence.length - 1;
      slider.max = String(maxIteration);
      slider.value = "0";
      render(0);
    }
    function renderCurve(selectedIteration) {
      convSvg.innerHTML = "";
      const w = 1000, h = 460, pad = { l: 80, r: 24, t: 20, b: 48 };
      const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
      const values = DATA.convergence;
      const scale = buildScale(values);
      const mappedValues = values.map(scale.yMap);
      const yMin = Math.min(...mappedValues), yMax = Math.max(...mappedValues), yRange = Math.max(yMax - yMin, 1e-9);
      const xScale = i => pad.l + (i / Math.max(maxIteration, 1)) * plotW;
      const yScale = v => pad.t + (1 - (scale.yMap(v) - yMin) / yRange) * plotH;
      convSvg.appendChild(svgEl("rect", { x: 0, y: 0, width: w, height: h, fill: "white" }));
      for (let i = 0; i <= 5; i++) {
        const y = pad.t + (i / 5) * plotH;
        const value = scale.tickValues[i];
        convSvg.appendChild(svgEl("line", { x1: pad.l, y1: y, x2: w - pad.r, y2: y, stroke: "#d7d2c8", "stroke-width": 1 }));
        convSvg.appendChild(svgEl("text", { x: 10, y: y + 4, fill: "#6b7280", "font-size": 12 }, fmt(value)));
      }
      let path = "";
      values.forEach((value, i) => { path += (i === 0 ? "M" : "L") + xScale(i) + " " + yScale(value) + " "; });
      convSvg.appendChild(svgEl("path", { d: path.trim(), fill: "none", stroke: "#1d4ed8", "stroke-width": 2.6 }));
      convSvg.appendChild(svgEl("line", { x1: xScale(selectedIteration), y1: pad.t, x2: xScale(selectedIteration), y2: h - pad.b, stroke: "#111827", "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
      convSvg.appendChild(svgEl("circle", { cx: xScale(selectedIteration), cy: yScale(values[selectedIteration]), r: 6, fill: "#111827" }));
    }
    function render(iteration) {
      $("functionValue").textContent = selectedFunction;
      $("iterationValue").textContent = String(iteration);
      $("fitnessValue").textContent = fmt(DATA.convergence[iteration]);
      $("finalValue").textContent = fmt(DATA.final_fitness);
      $("scaleValue").textContent = buildScale(DATA.convergence).label;
      renderCurve(iteration);
    }
    Object.keys(ALL_DATA.functions).forEach(name => { const option = document.createElement("option"); option.value = name; option.textContent = name; functionSelect.appendChild(option); });
    functionSelect.addEventListener("change", () => setFunction(functionSelect.value));
    slider.addEventListener("input", () => render(Number(slider.value)));
    $("prevBtn").addEventListener("click", () => { slider.value = String(Math.max(0, Number(slider.value) - 1)); render(Number(slider.value)); });
    $("nextBtn").addEventListener("click", () => { slider.value = String(Math.min(maxIteration, Number(slider.value) + 1)); render(Number(slider.value)); });
    setFunction(Object.keys(ALL_DATA.functions)[0]);
  </script>
</body>
</html>"""
    return (
        html.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
        .replace("__PAGE_TITLE__", page_title)
        .replace("__CHART_TITLE__", chart_title)
        .replace("__HINT_TEXT__", hint_text)
    )
