function normalizeX(n) {
  return map(n, 0, width, -1, 1)
}

function normalizeY(n) {
  return map(n, 0, height, 1, -1)
}

function denormalizeX(n) {
  return map(n, -1, 1, 0, width)
}

function denormalizeY(n) {
  return map(n, 1, -1, 0, height)
}
