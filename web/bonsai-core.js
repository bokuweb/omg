// Keep caller text literal: only renderer-owned segments may become control tokens.
export function packSegments(segments, tokenize) {
  const ids = [];
  for (const s of segments) {
    if (s.kind !== "text" && s.kind !== "special") throw Error(`Unsupported Bonsai segment: ${s.kind}`);
    const part = tokenize(s.value, s.kind === "special");
    if (s.kind === "special" && part.length !== 1) throw Error(`Bonsai control token is not atomic: ${s.value}`);
    for (const id of part) ids.push(id);
  }
  return ids;
}

export function preparePlan(plan, tokenize, cap, context = 2048) {
  const prefix = packSegments(plan.prefix, tokenize);
  if (!prefix.length || prefix.length >= context) throw Error(`State must fit the ${context.toLocaleString("en-US")}-token Bonsai context`);
  const branches = plan.branches.map(branch => {
    if (!branch.keys.length || branch.keys.length > cap) throw Error("Unsupported Bonsai option count");
    const ids = packSegments(branch.segments, tokenize);
    if (!ids.length || prefix.length + ids.length > context) throw Error(`State + question exceeds the ${context.toLocaleString("en-US")}-token Bonsai context; shorten the input`);
    return { ids, count: branch.keys.length };
  });
  return { prefix, branches };
}
