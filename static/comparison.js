(() => {
  const root = document.querySelector("[data-comparison-root]");
  const dataNode = document.getElementById("comparison-data");
  if (!root || !dataNode) return;

  const data = JSON.parse(dataNode.textContent);
  const styles = data.styles || [];
  const emotions = data.emotions || [];
  const emotionLabels = {
    anger: "Anger",
    disgust: "Disgust",
    fear: "Fear",
    happiness: "Happiness",
    sadness: "Sadness",
    surprise: "Surprise",
    neutral: "Neutral",
  };
  const activeStyles = new Set(styles.map((style) => style.id));
  const svgNamespace = "http://www.w3.org/2000/svg";

  function percent(value, digits = 1) {
    return `${(Number(value || 0) * 100).toFixed(digits)}%`;
  }

  function svgElement(name, attributes = {}, text = "") {
    const element = document.createElementNS(svgNamespace, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
    if (text) element.textContent = text;
    return element;
  }

  function visibleStyles() {
    return styles.filter((style) => activeStyles.has(style.id));
  }

  function renderEmotionChart() {
    const host = root.querySelector("[data-emotion-chart]");
    const detail = root.querySelector("[data-emotion-detail]");
    if (!host) return;
    host.replaceChildren();
    const shown = visibleStyles();
    if (!shown.length) {
      host.textContent = "Select at least one style to display the chart.";
      return;
    }

    const width = 920;
    const height = 380;
    const margin = { top: 18, right: 20, bottom: 58, left: 58 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const maximum = Math.max(...styles.flatMap((style) => emotions.map((emotion) => style.emotion_means[emotion] || 0)));
    const yMaximum = Math.max(0.1, Math.ceil(maximum * 10) / 10);
    const svg = svgElement("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": "Grouped bar chart comparing mean emotion scores by visual style",
    });
    svg.appendChild(svgElement("title", {}, "Mean emotion scores by style"));

    for (let tick = 0; tick <= 4; tick += 1) {
      const value = (yMaximum / 4) * tick;
      const y = margin.top + plotHeight - (value / yMaximum) * plotHeight;
      svg.appendChild(svgElement("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "chart-gridline" }));
      svg.appendChild(svgElement("text", { x: margin.left - 10, y: y + 4, "text-anchor": "end", class: "chart-axis-label" }, percent(value, 0)));
    }

    const categoryWidth = plotWidth / emotions.length;
    const groupWidth = categoryWidth * 0.72;
    const gap = 4;
    const barWidth = Math.max(8, (groupWidth - gap * (shown.length - 1)) / shown.length);

    emotions.forEach((emotion, emotionIndex) => {
      const categoryX = margin.left + emotionIndex * categoryWidth;
      const startX = categoryX + (categoryWidth - groupWidth) / 2;
      svg.appendChild(svgElement("text", {
        x: categoryX + categoryWidth / 2,
        y: height - 25,
        "text-anchor": "middle",
        class: "chart-axis-label",
      }, emotionLabels[emotion] || emotion));

      shown.forEach((style, styleIndex) => {
        const value = Number(style.emotion_means[emotion] || 0);
        const barHeight = (value / yMaximum) * plotHeight;
        const bar = svgElement("rect", {
          x: startX + styleIndex * (barWidth + gap),
          y: margin.top + plotHeight - barHeight,
          width: barWidth,
          height: barHeight,
          rx: 3,
          class: `emotion-bar style-fill-${style.id}`,
          tabindex: 0,
          role: "img",
          "aria-label": `${style.label}, ${emotionLabels[emotion]}: ${percent(value)}`,
        });
        const showDetail = () => {
          if (detail) detail.textContent = `${style.label} · ${emotionLabels[emotion]}: ${percent(value)}`;
        };
        bar.addEventListener("mouseenter", showDetail);
        bar.addEventListener("focus", showDetail);
        svg.appendChild(bar);
      });
    });
    host.appendChild(svg);
  }

  function renderMixChart() {
    const host = root.querySelector("[data-mix-chart]");
    const detail = root.querySelector("[data-mix-detail]");
    if (!host) return;
    host.replaceChildren();
    const shown = visibleStyles();
    if (!shown.length) {
      host.textContent = "Select at least one style to display the chart.";
      return;
    }

    shown.forEach((style) => {
      const row = document.createElement("div");
      row.className = "mix-row";
      const label = document.createElement("strong");
      label.textContent = style.label;
      const bar = document.createElement("div");
      bar.className = "mix-bar";
      bar.setAttribute("aria-label", `${style.label} top predicted emotion distribution`);

      emotions.forEach((emotion, index) => {
        const item = style.top_emotions[emotion] || { count: 0, rate: 0 };
        const segment = document.createElement("span");
        segment.className = `mix-segment emotion-fill-${index + 1}`;
        segment.style.width = `${item.rate * 100}%`;
        segment.tabIndex = 0;
        segment.setAttribute("role", "img");
        segment.setAttribute("aria-label", `${emotionLabels[emotion]}: ${item.count} faces, ${percent(item.rate)}`);
        const showDetail = () => {
          if (detail) detail.textContent = `${style.label} · ${emotionLabels[emotion]}: ${item.count} faces (${percent(item.rate)})`;
        };
        segment.addEventListener("mouseenter", showDetail);
        segment.addEventListener("focus", showDetail);
        bar.appendChild(segment);
      });
      row.append(label, bar);
      host.appendChild(row);
    });

    const legend = root.querySelector("[data-emotion-legend]");
    if (legend && !legend.children.length) {
      emotions.forEach((emotion, index) => {
        const item = document.createElement("span");
        item.innerHTML = `<i class="emotion-fill-${index + 1}" aria-hidden="true"></i>${emotionLabels[emotion]}`;
        legend.appendChild(item);
      });
    }
  }

  function allActionUnits() {
    return [...new Set(styles.flatMap((style) => Object.keys(style.au_values || {})))];
  }

  function auStatistics(code) {
    const values = styles.map((style) => Number(style.au_values[code] || 0));
    return {
      code,
      values,
      average: values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length),
      gap: Math.max(...values) - Math.min(...values),
    };
  }

  function renderAuChart() {
    const host = root.querySelector("[data-au-chart]");
    const order = root.querySelector("[data-au-order]")?.value || "gap";
    if (!host) return;
    host.replaceChildren();
    const shown = visibleStyles();
    if (!shown.length) {
      host.textContent = "Select at least one style to display the chart.";
      return;
    }

    const statistics = allActionUnits().map(auStatistics);
    statistics.sort((left, right) => {
      if (order === "code") return left.code.localeCompare(right.code, undefined, { numeric: true });
      return right[order] - left[order];
    });

    statistics.forEach(({ code }) => {
      const row = document.createElement("div");
      row.className = "au-comparison-row";
      const label = document.createElement("div");
      label.className = "au-comparison-label";
      label.innerHTML = `<strong>${code}</strong><span>${data.au_descriptions[code] || "Action unit"}</span>`;
      const bars = document.createElement("div");
      bars.className = "au-style-bars";

      shown.forEach((style) => {
        const value = Number(style.au_values[code] || 0);
        const line = document.createElement("div");
        line.className = "au-style-line";
        line.setAttribute("aria-label", `${code}, ${style.label}: ${percent(value)}`);
        line.innerHTML = `
          <span>${style.label}</span>
          <div><i class="style-fill-${style.id}" style="width:${value * 100}%"></i></div>
          <b>${percent(value, 0)}</b>
        `;
        bars.appendChild(line);
      });
      row.append(label, bars);
      host.appendChild(row);
    });
  }

  function renderAll() {
    renderEmotionChart();
    renderMixChart();
    renderAuChart();
  }

  root.querySelectorAll("[data-style-toggle]").forEach((toggle) => {
    toggle.addEventListener("change", () => {
      if (toggle.checked) activeStyles.add(toggle.value);
      else activeStyles.delete(toggle.value);
      renderAll();
    });
  });
  root.querySelector("[data-au-order]")?.addEventListener("change", renderAuChart);
  renderAll();
})();
