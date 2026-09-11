/**
 * Hide finger / palm joints in the Plotly skeleton so they do not occlude
 * the waist. Wrist endpoints (LeftHand / LeftWrist / …) stay visible.
 *
 * Matches HiPHI (LeftHandThumb1…), FZ (LeftHandPalm, FZLeftIndex1…),
 * and typical Xsens finger names.
 */
(function (global) {
  const HAND_JOINT_RE = /(thumb|index|middle|ring|pinky|finger|palm|carpal|inhand)/i;

  function isHiddenHumanHandJoint(name) {
    return typeof name === "string" && HAND_JOINT_RE.test(name);
  }

  function filterHumanSkeletonForVis(nodes, edges) {
    const keepNodes = (nodes || []).filter((n) => !isHiddenHumanHandJoint(n && n.name));
    const names = new Set(keepNodes.map((n) => n.name));
    const keepEdges = (edges || []).filter(
      (pair) => pair && names.has(pair[0]) && names.has(pair[1]),
    );
    return { nodes: keepNodes, edges: keepEdges };
  }

  global.isHiddenHumanHandJoint = isHiddenHumanHandJoint;
  global.filterHumanSkeletonForVis = filterHumanSkeletonForVis;
})(window);
