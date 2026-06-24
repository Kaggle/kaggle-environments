import svgSymbolPath from '../assets/icons.svg?url';
import geminiLogoPath from '../assets/gemini.svg?url';
import { getAgentBrand } from '../utils/agentLogos.ts';
import styles from './Playerbar.module.css';

interface Props {
  name?: string;
}

export function BrandLogo({ name }: Props) {
  if (!name) return null;
  const brand = getAgentBrand(name);

  // There is a bug on MacOS (and iOS) 26+, where `url()` inside svg
  // symbol sets don't work. We need this to render Gemini's gradient. So until
  // that's resolved we have to render an image as a special case for svg's
  // with gradients in them.
  if (brand === 'gemini') {
    return <img src={geminiLogoPath} className={styles.brandLogo} width="128" height="128" alt="" aria-hidden="true" />;
  }

  return (
    <svg
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      width="128"
      height="128"
      viewBox="0 0 128 128"
      className={styles.brandLogo}
    >
      <use href={`${svgSymbolPath}#${brand}`} />
    </svg>
  );
}
