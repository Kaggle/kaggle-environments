import iconsSvg from '../assets/icons.svg?raw';

// Inline the sprite so `<use href="#id">` works without fetching an asset.
// In production the bundle is loaded via `srcdoc` (document URL `about:srcdoc`),
// so relative asset URLs like `./assets/icons-HASH.svg` cannot be fetched.
export default function SvgSprite() {
  return <div className="visually-hidden" aria-hidden="true" dangerouslySetInnerHTML={{ __html: iconsSvg }} />;
}
