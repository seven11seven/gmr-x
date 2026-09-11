/**
 * Left → right name swap, same rules as Stage 1 mapping.
 * left_* / *_left_* / LeftXxx / LEFT_*  →  right / Right / RIGHT
 */
(function (global) {
  function hasLeftToken(name) {
    return /(?:^|_)left(?=_|$)|Left|LEFT/.test(name || "");
  }

  function swapSide(name, toRight) {
    const a = toRight ? "Left" : "Right";
    const b = toRight ? "Right" : "Left";
    const aa = toRight ? "LEFT" : "RIGHT";
    const bb = toRight ? "RIGHT" : "LEFT";
    const aaa = toRight ? "left" : "right";
    const bbb = toRight ? "right" : "left";
    return String(name).split(a).join(b).split(aa).join(bb).split(aaa).join(bbb);
  }

  function weightRow(robotName) {
    if (!robotName) return null;
    return document.querySelector(
      `#weightList .weight-row[data-robot="${CSS.escape(robotName)}"]`,
    );
  }

  function copyWeights(fromEl, toEl) {
    if (!fromEl || !toEl) return false;
    const pos = fromEl.querySelector(".pos-w");
    const rot = fromEl.querySelector(".rot-w");
    const tPos = toEl.querySelector(".pos-w");
    const tRot = toEl.querySelector(".rot-w");
    if (!pos || !rot || !tPos || !tRot) return false;
    tPos.value = pos.value;
    tRot.value = rot.value;
    return true;
  }

  function bindLiveLeftToRight(rowEl) {
    if (!hasLeftToken(rowEl.dataset.robot)) return;
    const rightName = swapSide(rowEl.dataset.robot, true);
    if (!rightName || rightName === rowEl.dataset.robot) return;
    rowEl.querySelectorAll(".pos-w, .rot-w").forEach((input) => {
      input.addEventListener("input", () => {
        const right = weightRow(rightName);
        if (!right) return;
        const cls = input.classList.contains("pos-w") ? ".pos-w" : ".rot-w";
        const dest = right.querySelector(cls);
        if (dest) dest.value = input.value;
      });
    });
  }

  function applyLeftToRightWeights() {
    const rows = [...document.querySelectorAll("#weightList .weight-row[data-robot]")];
    const byRobot = new Map(rows.map((el) => [el.dataset.robot, el]));
    let copied = 0;
    const missing = [];
    for (const el of rows) {
      if (!hasLeftToken(el.dataset.robot)) continue;
      const rightName = swapSide(el.dataset.robot, true);
      if (!rightName || rightName === el.dataset.robot) continue;
      const right = byRobot.get(rightName);
      if (!right) {
        missing.push(rightName);
        continue;
      }
      if (copyWeights(el, right)) copied += 1;
    }
    return { copied, missing };
  }

  global.GmrWeightMirror = {
    hasLeftToken,
    swapSide,
    bindLiveLeftToRight,
    applyLeftToRightWeights,
  };
})(window);
