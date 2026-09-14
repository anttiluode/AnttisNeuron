const stageCopy = {
  synapse: "Oja-style local learning can recover a leading covariance direction. AnttisNeuron reproduces that as a sanity gate; it does not claim that one isolated synapse independently performs PCA.",
  branch: "The synthetic cable has physical modes and timescales. Gate 5 shows that local edge activity can reshape this operator in one constructed world; Gate 6 shows the same rule is not universally beneficial.",
  nonlinearity: "This part is algebraically clean: if every branch remains static and linear, the whole tree collapses to one effective linear filter. A nonlinearity before pooling creates real hidden subunits.",
  ais: "The AIS is established biology as an action-potential initiation region, but AnttisNeuron has not yet earned the stronger co-adaptation story. Load compensation is a future falsification target on main."
};

for (const button of document.querySelectorAll("[data-stage]")) {
  button.addEventListener("click", () => {
    for (const peer of document.querySelectorAll("[data-stage]")) peer.classList.remove("active");
    button.classList.add("active");
    const detail = document.querySelector("#stage-detail");
    if (detail) detail.textContent = stageCopy[button.dataset.stage] || "";
  });
}

const receipts = {
  gate2: "results/gate2.json",
  gate4: "results/gate4.json",
  gate5: "results/gate5.json",
  gate5b: "results/gate5b.json",
  gate6: "results/gate6.json"
};

function readPath(object, path) {
  return path.split(".").reduce((value, key) => value == null ? undefined : value[key], object);
}

function formatNumber(value, kind, data) {
  if (!Number.isFinite(value)) return null;
  if (kind === "int") return String(Math.round(value));
  if (kind === "fraction-count") {
    const worlds = Number(data.n_worlds);
    if (!Number.isFinite(worlds)) return null;
    return `${Math.round(value * worlds)} / ${Math.round(worlds)}`;
  }
  if (kind === "signed") return `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

async function loadReceipt(name, url) {
  const card = document.querySelector(`[data-receipt-card="${name}"]`);
  if (!card) return;
  const state = card.querySelector(".receipt-state");
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    let incomplete = false;
    for (const slot of card.querySelectorAll("[data-value]")) {
      const value = readPath(data, slot.dataset.value);
      const rendered = formatNumber(Number(value), slot.dataset.format || "number", data);
      if (rendered == null) {
        slot.textContent = "—";
        incomplete = true;
      } else {
        slot.textContent = rendered;
      }
    }
    if (incomplete) {
      state.textContent = "receipt incomplete";
      state.classList.add("bad");
    } else {
      state.textContent = "receipt verified";
      state.classList.add("ok");
    }
  } catch (error) {
    state.textContent = "receipt unavailable";
    state.classList.add("bad");
    card.dataset.receiptError = String(error);
  }
}

for (const [name, url] of Object.entries(receipts)) loadReceipt(name, url);
