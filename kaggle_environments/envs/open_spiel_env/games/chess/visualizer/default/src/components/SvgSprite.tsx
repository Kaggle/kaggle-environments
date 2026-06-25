import iconsSvg from '../assets/icons.svg?raw';

// Inline the sprite into the document so `<use href="#id">` resolves without
// fetching an asset URL. The production build is bundled into a single HTML
// served from an iframe `srcdoc`, where relative asset URLs have no base to
// resolve against and silently fail to load.
export default function SvgSprite() {
  return (
    <div
      aria-hidden="true"
      style={{ width: 0, height: 0, position: 'absolute', overflow: 'hidden' }}
      dangerouslySetInnerHTML={{ __html: iconsSvg }}
    />
  );
}
