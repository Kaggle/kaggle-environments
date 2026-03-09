export function UiSvgFilter() {
  return (
    <svg>
      <defs>
        <filter
          id="ui-displacement"
          x="-0.00164473"
          y="-0.000179887"
          width="90.3158"
          height="90.3158"
          filterUnits="userSpaceOnUse"
          color-interpolation-filters="sRGB"
        >
          <feFlood flood-opacity="0" result="BackgroundImageFix" />
          <feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape" />
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.061688315123319626 0.061688315123319626"
            numOctaves="3"
            seed="2775"
          />
          <feDisplacementMap
            in="shape"
            scale="2.3157894611358643"
            xChannelSelector="R"
            yChannelSelector="G"
            result="displacedImage"
            width="100%"
            height="100%"
          />
          <feMerge result="effect1_texture_222_21018">
            <feMergeNode in="displacedImage" />
          </feMerge>
        </filter>
      </defs>
    </svg>
  );
}
