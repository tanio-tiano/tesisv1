from pathlib import Path
import json

import pandas as pd
from pandas.errors import EmptyDataError


FEATURE_NAMES = [
    "alpha",
    "beta",
    "danger_signal",
    "safety_signal",
    "diversity",
    "diversity_norm",
    "iteration",
]


def create_all_functions_monitor(values_dir, graphs_dir):
    values_dir = Path(values_dir)
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    functions = {}
    for function_id in range(1, 13):
        payload = _load_payload(values_dir, function_id)
        if payload is not None:
            functions[f"F{function_id}"] = payload

    if not functions:
        return None

    html = _build_html({"feature_names": FEATURE_NAMES, "functions": functions})
    monitor_path = graphs_dir / "interactive_monitor_all_functions.html"
    monitor_path.write_text(html, encoding="utf-8")
    return monitor_path


def _load_payload(values_dir, function_id):
    curve_path = values_dir / f"conv_curve_shap_F{function_id}.csv"
    base_curve_path = (
        Path(__file__).resolve().parents[1]
        / "WO_base_cec"
        / "all_functions_outputs_final"
        / "values"
        / f"conv_curve_F{function_id}.csv"
    )
    state_path = values_dir / f"controller_state_F{function_id}.csv"
    shap_path = values_dir / f"shap_values_F{function_id}.csv"
    event_path = values_dir / f"controller_events_F{function_id}.csv"
    if not curve_path.exists() or not state_path.exists() or not shap_path.exists():
        return None

    curve_df = pd.read_csv(curve_path)
    state_df = pd.read_csv(state_path)
    try:
        shap_df = pd.read_csv(shap_path)
    except EmptyDataError:
        shap_df = pd.DataFrame()
    try:
        event_df = pd.read_csv(event_path) if event_path.exists() else pd.DataFrame()
    except EmptyDataError:
        event_df = pd.DataFrame()
    if curve_df.empty or state_df.empty:
        return None

    states = []
    for _, row in state_df.iterrows():
        states.append(
            {
                "iteration": int(row["iteration"]),
                "event_id": str(row.get("event_id", "0")),
                "alpha": _number(row["alpha"]),
                "beta": _number(row["beta"]),
                "danger_signal": _number(row["danger_signal"]),
                "safety_signal": _number(row["safety_signal"]),
                "diversity": _number(row.get("diversity", 0.0)),
                "diversity_norm": _number(row.get("diversity_norm", 0.0)),
                "current_fitness": _number(row["current_fitness"]),
                "output_fitness": _number(row["output_fitness"]),
                "stagnation": int(row.get("stagnation", row.get("event_active", 0))),
                "stagnation_length": int(row.get("stagnation_length", row.get("s_t", 0))),
                "diversity": _number(row.get("diversity", 0.0)),
                "diversity_norm": _number(row.get("diversity_norm", 0.0)),
                "diagnosis": str(row.get("diagnosis", row.get("stagnation_state", "no_stagnation"))),
                "s_t": int(row["s_t"]),
                "delta_fitness": _number(row["delta_fitness"]),
                "threshold": _number(row["threshold"]),
                "stagnation_state": str(row["stagnation_state"]),
                "event_active": int(row["event_active"]),
                "control_mode": str(row["control_mode"]),
                "action_taken": str(row["action_taken"]),
                "action_justification": str(row["action_justification"]),
                "alpha_scale": _number(row.get("alpha_scale", 1.0)),
                "beta_scale": _number(row["beta_scale"]),
                "rescue_fraction": _number(row["rescue_fraction"]),
                "rescue_mode": str(row.get("rescue_mode", "none")),
                "rescued_agents": int(row["rescued_agents"]),
                "shap_requested": int(row["shap_requested"]),
                "shap_reason": str(row["shap_reason"]),
                "dominant_feature": str(row["dominant_feature"]),
                "trace_role": str(row.get("trace_role", "none")),
                "effect_label": str(row.get("effect_label", "not_applicable")),
                "delta_fitness_post": _number(row.get("delta_fitness_post", 0.0)),
            }
        )

    shap_rows = {}
    for _, row in shap_df.iterrows():
        iteration = int(row["iteration"])
        shap_rows.setdefault(iteration, []).append(
            {
                "event_id": int(row.get("event_id", 0)),
                "phase": str(row.get("phase", "episode")),
                "iteration": iteration,
                "t_pre": int(row.get("t_pre", iteration)),
                "t_post": int(row.get("t_post", iteration)),
                "event_active": int(row["event_active"]),
                "shap_reason": str(row.get("shap_reason", "")),
                "selected_output": str(row["selected_output"]),
                "base_value": _number(row["BASE_fitness"]),
                "output_value": _number(row["OUTPUT_fitness"]),
                "observed_value": _number(row.get("OBSERVED_fitness", row["OUTPUT_fitness"])),
                "stagnation_state": str(row.get("stagnation_state", "")),
                "action_taken": str(row.get("action_taken", "")),
                "dominant_feature": str(row["dominant_feature"]),
                "contributions": {
                    feature: _number(row.get(f"SHAP_{feature}", float("nan")))
                    for feature in FEATURE_NAMES
                },
            }
        )

    event_iterations = []
    if not event_df.empty and "iteration" in event_df.columns:
        event_iterations = [int(v) for v in event_df["iteration"].tolist()]
    event_trace = []
    if not event_df.empty:
        for _, row in event_df.iterrows():
            event_trace.append(
                {
                    "event_id": int(row.get("event_id", 0)),
                    "iteration": int(row["iteration"]),
                    "t_pre": int(row.get("t_pre", row["iteration"])),
                    "t_action": int(row.get("t_action", row["iteration"])),
                    "t_post": int(row.get("t_post", row["iteration"])),
                    "action_taken": str(row.get("action_taken", "")),
                    "stagnation_state": str(row.get("stagnation_state", "")),
                    "dominant_feature_pre": str(row.get("dominant_feature_pre", "none")),
                    "dominant_feature_post": str(row.get("dominant_feature_post", "none")),
                    "fitness_pre": _number(row.get("fitness_pre", float("nan"))),
                    "fitness_post": _number(row.get("fitness_post", float("nan"))),
                    "delta_fitness_post": _number(row.get("delta_fitness_post", 0.0)),
                    "effective_intervention": int(row.get("effective_intervention", 0)),
                    "effect_label": str(row.get("effect_label", "not_available")),
                    "diversity_pre": _number(row.get("diversity_pre", float("nan"))),
                    "diversity_post": _number(row.get("diversity_post", float("nan"))),
                    "diversity_norm_pre": _number(row.get("diversity_norm_pre", float("nan"))),
                    "diversity_norm_post": _number(row.get("diversity_norm_post", float("nan"))),
                }
            )

    convergence = [_number(v) for v in curve_df["best_fitness"].tolist()]
    base_convergence = []
    if base_curve_path.exists():
        try:
            base_curve_df = pd.read_csv(base_curve_path)
            if not base_curve_df.empty:
                base_convergence = [_number(v) for v in base_curve_df["best_fitness"].tolist()]
        except Exception:
            base_convergence = []
    return {
        "function_id": int(function_id),
        "convergence": convergence,
        "base_convergence": base_convergence,
        "states": states,
        "shap_rows": shap_rows,
        "event_iterations": event_iterations,
        "event_trace": event_trace,
        "event_count": len(event_iterations),
        "shap_count": int(sum(len(rows) for rows in shap_rows.values())),
        "final_fitness": convergence[-1],
        "base_final_fitness": base_convergence[-1] if base_convergence else None,
    }


def _number(value):
    value = float(value)
    if value == float("inf"):
        return "inf"
    if value == -float("inf"):
        return "-inf"
    return value


def _build_html(payload):
    html = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Monitor WO SHAP</title>
  <style>
    :root { --panel:#fffaf0; --ink:#1f2937; --muted:#6b7280; --grid:#d7d2c8; --line:#1d4ed8; --event:#b45309; --pos:#c2410c; --neg:#0f766e; --accent:#111827; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Segoe UI", Tahoma, sans-serif; background: linear-gradient(180deg, #f7f3ea 0%, #efe8d8 100%); color: var(--ink); }
    .wrap { max-width: 1600px; margin: 0 auto; padding: 20px; }
    .topbar { display: grid; grid-template-columns: 1fr auto auto auto auto; gap: 12px; align-items: center; margin-bottom: 18px; }
    .title { font-size: 26px; font-weight: 700; }
    .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    button, select { border: 0; background: var(--accent); color: white; padding: 10px 14px; cursor: pointer; font-size: 14px; border-radius: 10px; }
    input[type="range"] { width: 360px; }
    .badge { background: rgba(17,24,39,0.08); border-radius: 999px; padding: 8px 12px; font-size: 13px; color: var(--muted); }
    .meta { display: grid; grid-template-columns: repeat(10, minmax(0,1fr)); gap: 10px; margin-bottom: 12px; }
    .grid { display: grid; grid-template-columns: 1.35fr 1fr; gap: 18px; }
    .panel { background: var(--panel); border: 1px solid rgba(0,0,0,0.06); border-radius: 18px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.06); }
    .panel h2 { margin: 0 0 8px 0; font-size: 18px; }
    .meta-card { background: rgba(255,255,255,0.7); border: 1px solid rgba(0,0,0,0.05); border-radius: 12px; padding: 10px 12px; }
    .meta-card .k { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .meta-card .v { font-size: 16px; font-weight: 700; }
    .info { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; margin-top: 12px; }
    svg { width: 100%; height: auto; display: block; background: white; border-radius: 14px; border: 1px solid rgba(0,0,0,0.05); }
    .hint { margin-top: 10px; font-size: 13px; color: var(--muted); }
    .legend { display: flex; gap: 16px; margin-top: 10px; flex-wrap: wrap; color: var(--muted); font-size: 13px; }
    .swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: middle; }
    @media (max-width: 1100px) { .grid,.topbar { grid-template-columns: 1fr; } input[type="range"] { width: 100%; } .meta { grid-template-columns: repeat(2, minmax(0,1fr)); } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="title">Monitor interactivo WO SHAP</div>
      <div class="controls"><select id="functionSelect"></select></div>
      <div class="controls"><button id="prevBtn">Anterior</button><button id="nextBtn">Siguiente</button></div>
      <div class="controls"><input id="iterSlider" type="range" min="0" max="0" value="0"></div>
      <div class="badge" id="iterBadge">Iteracion 0</div>
    </div>
    <div class="meta">
      <div class="meta-card"><div class="k">Fitness actual</div><div class="v" id="currentFitnessValue">-</div></div>
      <div class="meta-card"><div class="k">Fitness salida</div><div class="v" id="outputFitnessValue">-</div></div>
      <div class="meta-card"><div class="k">Fitness base final</div><div class="v" id="baseFinalFitnessValue">-</div></div>
      <div class="meta-card"><div class="k">Estado</div><div class="v" id="stagnationState">-</div></div>
      <div class="meta-card"><div class="k">s_t</div><div class="v" id="stagnationCount">-</div></div>
      <div class="meta-card"><div class="k">Diversidad norm.</div><div class="v" id="diversityNormValue">-</div></div>
      <div class="meta-card"><div class="k">Delta fitness</div><div class="v" id="deltaFitnessValue">-</div></div>
      <div class="meta-card"><div class="k">Accion</div><div class="v" id="actionTaken">-</div></div>
      <div class="meta-card"><div class="k">Eventos / SHAP</div><div class="v" id="eventSummary">-</div></div>
      <div class="meta-card"><div class="k">Escala grafico</div><div class="v" id="scaleValue">-</div></div>
      <div class="meta-card"><div class="k">Efecto interv.</div><div class="v" id="effectValue">-</div></div>
    </div>
    <div class="grid">
      <div class="panel">
        <h2>Curva de convergencia</h2>
        <svg id="convSvg" viewBox="0 0 860 440" preserveAspectRatio="none"></svg>
        <div class="legend"><span><span class="swatch" style="background:#1d4ed8"></span>Curva controlada</span><span><span class="swatch" style="background:#64748b"></span>Curva base</span><span><span class="swatch" style="background:#b45309"></span>Intervenciones</span><span><span class="swatch" style="background:#111827"></span>Seleccionada</span></div>
        <div class="hint" id="actionJustification">-</div>
      </div>
      <div class="panel">
        <h2>Contribuciones SHAP</h2>
        <svg id="shapSvg" viewBox="0 0 760 440" preserveAspectRatio="none"></svg>
        <div class="info">
          <div class="meta-card"><div class="k">Output SHAP</div><div class="v" id="outputValue">-</div></div>
          <div class="meta-card"><div class="k">Base</div><div class="v" id="baseValue">-</div></div>
          <div class="meta-card"><div class="k">Feature dominante</div><div class="v" id="dominantFeature">-</div></div>
          <div class="meta-card"><div class="k">Motivo SHAP</div><div class="v" id="shapReason">-</div></div>
        </div>
        <div class="hint" id="shapHint">-</div>
      </div>
    </div>
    <div class="panel" style="margin-top:18px">
      <h2>Traza de intervencion</h2>
      <div class="info">
        <div class="meta-card"><div class="k">Evento</div><div class="v" id="traceEventId">-</div></div>
        <div class="meta-card"><div class="k">Fase SHAP</div><div class="v" id="tracePhase">-</div></div>
        <div class="meta-card"><div class="k">t_pre / t_post</div><div class="v" id="traceWindow">-</div></div>
        <div class="meta-card"><div class="k">Delta post</div><div class="v" id="traceDelta">-</div></div>
        <div class="meta-card"><div class="k">Fitness pre / post</div><div class="v" id="traceFitness">-</div></div>
        <div class="meta-card"><div class="k">Dominante pre / post</div><div class="v" id="traceDominance">-</div></div>
      </div>
      <div class="hint" id="traceHint">Selecciona una iteracion asociada a una intervencion para ver su trazabilidad.</div>
    </div>
  </div>
  <script>
    const ALL_DATA = __PAYLOAD__;
    let DATA = null, selectedFunction = null, statesByIteration = new Map(), shapByIteration = new Map(), eventTraceById = new Map(), maxIteration = 0;
    const $ = id => document.getElementById(id);
    const functionSelect = $("functionSelect"), slider = $("iterSlider"), convSvg = $("convSvg"), shapSvg = $("shapSvg");
    function numeric(v) { if (v === "inf") return Infinity; if (v === "-inf") return -Infinity; return Number(v); }
    function fmt(v) {
      const n = numeric(v);
      if (!Number.isFinite(n)) return "inf";
      if (Math.abs(n) >= 1e4 || (Math.abs(n) > 0 && Math.abs(n) < 1e-3)) return n.toExponential(3);
      return n.toFixed(6);
    }
    function svgEl(name, attrs = {}, text = null) { const el = document.createElementNS("http://www.w3.org/2000/svg", name); for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v); if (text !== null) el.textContent = text; return el; }
    function buildScale(values) {
      const finiteValues = values.filter(v => Number.isFinite(v));
      const yMinRaw = Math.min(...finiteValues), yMaxRaw = Math.max(...finiteValues);
      const positiveValues = finiteValues.filter(v => v > 0);
      const safeMin = positiveValues.length ? Math.max(Math.min(...positiveValues), 1e-12) : 1e-12;
      const positiveOnly = finiteValues.every(v => v > 0);
      const ratio = positiveOnly ? (yMaxRaw / safeMin) : 1;
      const useLog = positiveOnly && ratio >= 1e3;
      if (useLog) {
        const logMin = Math.log10(safeMin);
        const logMax = Math.log10(Math.max(yMaxRaw, safeMin * 1.000001));
        return {
          mode: "log10",
          label: "log10",
          yMap: v => Math.log10(Math.max(v, safeMin)),
          tickValues: Array.from({ length: 6 }, (_, i) => Math.pow(10, logMax - (i / 5) * (logMax - logMin))),
        };
      }
      const yRange = Math.max(yMaxRaw - yMinRaw, 1e-9);
      return {
        mode: "linear",
        label: "lineal",
        yMap: v => v,
        tickValues: Array.from({ length: 6 }, (_, i) => yMaxRaw - (i / 5) * yRange),
      };
    }
    function setFunction(name) {
      selectedFunction = name; DATA = ALL_DATA.functions[name];
      statesByIteration = new Map(DATA.states.map(row => [row.iteration, row]));
      shapByIteration = new Map(Object.entries(DATA.shap_rows).map(([k, v]) => [Number(k), v]));
      eventTraceById = new Map((DATA.event_trace || []).map(row => [Number(row.event_id), row]));
      maxIteration = DATA.convergence.length - 1; slider.max = String(maxIteration); slider.value = "0";
      $("eventSummary").textContent = `${DATA.event_count} / ${DATA.shap_count}`; render(0);
    }
    function renderConvergence(selectedIteration) {
      convSvg.innerHTML = ""; const w = 860, h = 440, pad = { l: 70, r: 20, t: 18, b: 42 };
      const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b, values = DATA.convergence.map(numeric), baseValues = (DATA.base_convergence || []).map(numeric);
      const allValues = values.concat(baseValues);
      const scale = buildScale(allValues);
      const mappedValues = allValues.map(scale.yMap);
      const yMin = Math.min(...mappedValues), yMax = Math.max(...mappedValues), yRange = Math.max(yMax - yMin, 1e-9);
      const xScale = i => pad.l + (i / Math.max(maxIteration, 1)) * plotW, yScale = v => pad.t + (1 - (scale.yMap(v) - yMin) / yRange) * plotH;
      convSvg.appendChild(svgEl("rect", { x: 0, y: 0, width: w, height: h, fill: "white" }));
      for (let i = 0; i <= 5; i++) { const y = pad.t + (i / 5) * plotH; convSvg.appendChild(svgEl("line", { x1: pad.l, y1: y, x2: w - pad.r, y2: y, stroke: "#d7d2c8", "stroke-width": 1 })); convSvg.appendChild(svgEl("text", { x: 8, y: y + 4, fill: "#6b7280", "font-size": 12 }, fmt(scale.tickValues[i]))); }
      if (baseValues.length > 0) {
        let basePath = ""; baseValues.forEach((v, i) => { basePath += (i === 0 ? "M" : "L") + xScale(i) + " " + yScale(v) + " "; });
        convSvg.appendChild(svgEl("path", { d: basePath.trim(), fill: "none", stroke: "#64748b", "stroke-width": 2.2, "stroke-dasharray": "7 5" }));
      }
      let path = ""; values.forEach((v, i) => { path += (i === 0 ? "M" : "L") + xScale(i) + " " + yScale(v) + " "; });
      convSvg.appendChild(svgEl("path", { d: path.trim(), fill: "none", stroke: "#1d4ed8", "stroke-width": 2.5 }));
      DATA.event_iterations.forEach(iter => convSvg.appendChild(svgEl("circle", { cx: xScale(iter), cy: yScale(values[iter]), r: 4.5, fill: "#b45309", stroke: "#111827", "stroke-width": 1 })));
      (DATA.event_trace || []).forEach(trace => {
        const postIter = Number(trace.t_post);
        if (postIter >= 0 && postIter < values.length) convSvg.appendChild(svgEl("circle", { cx: xScale(postIter), cy: yScale(values[postIter]), r: 4, fill: "#16a34a", stroke: "#111827", "stroke-width": 1 }));
      });
      convSvg.appendChild(svgEl("line", { x1: xScale(selectedIteration), y1: pad.t, x2: xScale(selectedIteration), y2: h - pad.b, stroke: "#111827", "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
      convSvg.appendChild(svgEl("circle", { cx: xScale(selectedIteration), cy: yScale(values[selectedIteration]), r: 6, fill: "#111827" }));
    }
    function renderShap(selectedIteration) {
      shapSvg.innerHTML = ""; const w = 760, h = 440, pad = { l: 160, r: 20, t: 24, b: 40 };
      const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b, shapRows = shapByIteration.get(selectedIteration) || [];
      const shapRow = shapRows.find(row => row.phase === "pre") || shapRows[0];
      shapSvg.appendChild(svgEl("rect", { x: 0, y: 0, width: w, height: h, fill: "white" }));
      if (!shapRow) { shapSvg.appendChild(svgEl("text", { x: 40, y: 80, fill: "#6b7280", "font-size": 18 }, "No hay explicacion SHAP exacta en esta iteracion.")); $("outputValue").textContent = "-"; $("baseValue").textContent = "-"; $("dominantFeature").textContent = "-"; $("shapReason").textContent = "-"; $("shapHint").textContent = "Selecciona una iteracion con SHAP exacto."; return; }
      const pairs = ALL_DATA.feature_names.map(name => [name, numeric(shapRow.contributions[name])]), maxAbs = Math.max(...pairs.map(([, v]) => Math.abs(v)), 1e-9);
      const xCenter = pad.l + plotW / 2, xScale = v => (Math.abs(v) / maxAbs) * (plotW / 2 - 24), barH = plotH / pairs.length * 0.6, rowGap = plotH / pairs.length;
      shapSvg.appendChild(svgEl("line", { x1: xCenter, y1: pad.t - 8, x2: xCenter, y2: h - pad.b, stroke: "#9ca3af", "stroke-width": 1.2, "stroke-dasharray": "5 4" }));
      pairs.forEach(([name, value], idx) => { const y = pad.t + idx * rowGap + rowGap / 2 - barH / 2, width = xScale(value), isPositive = value >= 0, x = isPositive ? xCenter : xCenter - width; shapSvg.appendChild(svgEl("text", { x: 16, y: y + barH / 2 + 4, fill: "#111827", "font-size": 14 }, name)); shapSvg.appendChild(svgEl("rect", { x, y, width, height: barH, fill: isPositive ? "#c2410c" : "#0f766e", rx: 4 })); shapSvg.appendChild(svgEl("text", { x: isPositive ? x + width + 8 : x - 8, y: y + barH / 2 + 4, fill: "#374151", "font-size": 13, "text-anchor": isPositive ? "start" : "end" }, value.toFixed(6))); });
      $("outputValue").textContent = fmt(shapRow.output_value); $("baseValue").textContent = fmt(shapRow.base_value); $("dominantFeature").textContent = shapRow.dominant_feature; $("shapReason").textContent = `${shapRow.shap_reason} (${shapRow.phase})`; $("shapHint").textContent = shapRows.length > 1 ? `Hay ${shapRows.length} explicaciones en esta iteracion; se muestra la prioridad pre.` : "Contribucion positiva empeora el fitness; contribucion negativa lo reduce.";
      return shapRow;
    }
    function renderTrace(selectedIteration, state, shapRow) {
      $("effectValue").textContent = state.effect_label || "-";
      if (!shapRow || !shapRow.event_id) {
        $("traceEventId").textContent = "-";
        $("tracePhase").textContent = "-";
        $("traceWindow").textContent = "-";
        $("traceDelta").textContent = "-";
        $("traceFitness").textContent = "-";
        $("traceDominance").textContent = "-";
        $("traceHint").textContent = "Esta iteracion no pertenece a una traza pre/post de intervencion.";
        return;
      }
      const trace = eventTraceById.get(Number(shapRow.event_id));
      if (!trace) {
        $("traceEventId").textContent = String(shapRow.event_id);
        $("tracePhase").textContent = shapRow.phase;
        $("traceWindow").textContent = `${shapRow.t_pre} / ${shapRow.t_post}`;
        $("traceDelta").textContent = "-";
        $("traceFitness").textContent = "-";
        $("traceDominance").textContent = "-";
        $("traceHint").textContent = "No se encontro el resumen del evento en el log.";
        return;
      }
      $("traceEventId").textContent = String(trace.event_id);
      $("tracePhase").textContent = shapRow.phase;
      $("traceWindow").textContent = `${trace.t_pre} / ${trace.t_post}`;
      $("traceDelta").textContent = fmt(trace.delta_fitness_post);
      $("traceFitness").textContent = `${fmt(trace.fitness_pre)} / ${fmt(trace.fitness_post)}`;
      $("traceDominance").textContent = `${trace.dominant_feature_pre} / ${trace.dominant_feature_post}`;
      $("traceHint").textContent = `Efecto ${trace.effect_label}. Delta post = fitness_pre - fitness_post; positivo indica mejora tras la ventana post.`;
    }
    function render(iteration) {
      const state = statesByIteration.get(iteration); $("iterBadge").textContent = `${selectedFunction} - Iteracion ${iteration}`;
      $("currentFitnessValue").textContent = fmt(state.current_fitness); $("outputFitnessValue").textContent = fmt(state.output_fitness); $("baseFinalFitnessValue").textContent = DATA.base_final_fitness === null ? "-" : fmt(DATA.base_final_fitness); $("stagnationState").textContent = state.stagnation_state; $("stagnationCount").textContent = String(state.s_t); $("diversityNormValue").textContent = fmt(state.diversity_norm); $("deltaFitnessValue").textContent = fmt(state.delta_fitness); $("actionTaken").textContent = state.action_taken; $("scaleValue").textContent = buildScale(DATA.convergence.map(numeric).concat((DATA.base_convergence || []).map(numeric))).label; $("actionJustification").textContent = state.action_justification; renderConvergence(iteration); const shapRow = renderShap(iteration); renderTrace(iteration, state, shapRow);
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
    return html.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
