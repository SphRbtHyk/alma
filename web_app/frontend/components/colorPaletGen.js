/**
 * Interpolates between two numbers
 * @param {Int} a Start value.
 * @param {Int} b End value.
 * @param {float} t Interpolation factor.
 * @returns The interpolated value.
 */
function lerp(a, b, t) {
  return a + (b - a) * t;
}

/**
 * Converts hex color (#RRGGBB) to RGB array [r, g, b].
 * @param {String} hex The string representation of the hex color to be
 * converted.
 * @returns Array containing the RGB colors ([r, g, b]).
 */
function hexToRgb(hex) {
  hex = hex.replace("#", "");
  if (hex.length === 3) {
    hex = hex.split("").map(c => c + c).join("");
  }
  const bigint = parseInt(hex, 16);
  return [(bigint >> 16) & 255, (bigint >> 8) & 255, bigint & 255];
}

/**
 * Converts RGB array [r,g,b] to hex (#RRGGBB).
 * @param {Array} rgbArray Array containg rgb values. 
 * @returns The string representation of the hex value for the converted rgb
 * values.
 */
function rgbToHex([r, g, b]) {
  return (
    "#" +
    [r, g, b]
      .map(x => {
        const hex = Math.round(x).toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
  );
}

/**
 * Generates a smooth color palette.
 * @param {string[]} colors Array containing string representations of hex
 * colors.
 * @param {number} n The number of colors in the generated collor palette.
 * @returns {string[]} Array containing string represetations of hex color
 * vlaues.
 */
export default function generatePalette(colors, n) {
  if (colors.length < 2) throw new Error("Need at least 2 input colors");

  const rgbColors = colors.map(hexToRgb);
  const segments = colors.length - 1;
  const colorsPerSegment = (n - 1) / segments;

  const output = [];

  for (let i = 0; i < segments; i++) {
    const c1 = rgbColors[i];
    const c2 = rgbColors[i + 1];

    for (let j = 0; j < colorsPerSegment; j++) {
      const t = j / colorsPerSegment;
      const r = lerp(c1[0], c2[0], t);
      const g = lerp(c1[1], c2[1], t);
      const b = lerp(c1[2], c2[2], t);
      output.push(rgbToHex([r, g, b]));
    }
  }

  output.push(colors[colors.length - 1]); // include last color
  return output;
}
