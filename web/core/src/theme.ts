import { createTheme, Theme } from '@mui/material/styles';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/700.css';

// Custom breakpoints matching the Material design guidelines
// See: https://carbon.googleplex.com/kaggle/pages/layout-breakpoints/principles
declare module '@mui/material/styles' {
  interface BreakpointOverrides {
    xs: true;
    sm: true;
    md: true;
    lg: true;
    xl: true;
    // Custom breakpoints
    xs1: true;
    xs2: true;
    xs3: true;
    sm1: true;
    sm2: true;
    sm3: true;
    md1: true;
    md2: true;
    lg1: true;
    lg2: true;
    lg3: true;
    xl1: true;
    // Aliases
    phone: true;
    tablet: true;
    desktop: true;
  }
}

// Extend MUI Button variants
declare module '@mui/material/Button' {
  interface ButtonPropsVariantOverrides {
    medium: true;
    low: true;
  }
}

// Color constants
const COLORS = {
  KAGGLE_BLACK: '#202124', // intentionally the same as GREY_900
  KAGGLE_WHITE: '#FFFFFF',
  KAGGLE_FOCUS: '#20BEFF',
  GREY_50: '#F8F9FA',
  GREY_100: '#F1F3F4',
  GREY_200: '#E8EAED',
  GREY_300: '#DADCE0',
  GREY_400: '#BDC1C6',
  GREY_500: '#9AA0A6',
  GREY_600: '#80868B',
  GREY_700: '#5F6368',
  GREY_800: '#3C4043',
  GREY_850: '#2E3033',
  GREY_900: '#202124',
  GREY_950: '#1C1D20',
};

export const themeBreakpoints = {
  values: {
    xs: 0,
    xs1: 360,
    xs2: 400,
    xs3: 480, // phone
    sm: 600,
    sm1: 600,
    sm2: 720,
    sm3: 840, // tablet
    md: 960,
    md1: 960,
    md2: 1024,
    lg: 1280,
    lg1: 1280, // desktop
    lg2: 1440,
    lg3: 1600,
    xl: 1920,
    xl1: 1920,
    // Aliases
    phone: 480,
    tablet: 840,
    desktop: 1280,
  },
};

const themeTypography = {
  fontFamily: 'Inter, sans-serif',
  h1: {
    fontSize: '36px',
    lineHeight: '44px',
    fontWeight: '700',
  },
  h2: {
    fontSize: '32px',
    lineHeight: '40px',
    fontWeight: '700',
  },
  h3: {
    fontSize: '28px',
    lineHeight: '36px',
    fontWeight: '700',
  },
  h4: {
    fontSize: '24px',
    lineHeight: '32px',
    fontWeight: '700',
  },
  h5: {
    fontSize: '20px',
    lineHeight: '24px',
    fontWeight: '700',
  },
  h6: {
    fontSize: '16px',
    lineHeight: '20px',
    fontWeight: '700',
  },
  body1: {
    fontSize: '16px',
    lineHeight: '24px',
    fontWeight: '400',
  },
  body2: {
    fontSize: '14px',
    lineHeight: '20px',
    fontWeight: '400',
  },
};

// Base theme with structural (non-color) styles
const baseTheme = createTheme({
  breakpoints: themeBreakpoints,
  typography: themeTypography,
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          WebkitFontSmoothing: 'auto',
          MozOsxFontSmoothing: 'auto',
        },
        // Keep icons with antialiased rendering
        '.MuiIcon-root, .MuiSvgIcon-root, .material-symbols-outlined': {
          WebkitFontSmoothing: 'antialiased',
          MozOsxFontSmoothing: 'grayscale',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: '20px',
          fontFamily: 'Inter',
          padding: '0px 16px 0px 12px',
          height: '36px',
          width: 'fit-content',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '14px',
          lineHeight: '20px',
          fontWeight: '700',
        },
        startIcon: {
          display: 'flex',
          alignItems: 'center',
          marginRight: '8px',
          marginLeft: 0,
        },
        endIcon: {
          display: 'flex',
          alignItems: 'center',
          marginLeft: '8px',
          marginRight: 0,
        },
        sizeSmall: {
          height: '28px',
          fontWeight: 400,
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        },
      },
    },
    MuiIcon: {
      styleOverrides: {
        root: {
          height: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        },
      },
    },
    MuiSvgIcon: {
      styleOverrides: {
        root: {
          display: 'block',
        },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          fontSize: '14px',
          height: '32px',
          '& .MuiIcon-root': {
            marginRight: '16px',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        select: {
          padding: '0 32px 0 0',
        },
      },
    },
    MuiSlider: {
      styleOverrides: {
        root: {
          height: '3px',
          borderRadius: 0,
          marginBottom: 0,
        },
        mark: {
          display: 'none',
        },
        markLabel: {
          top: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 2,
        },
        valueLabel: {
          borderRadius: '25px',
          padding: '4px 8px',
          ':before': {
            display: 'none',
          },
        },
      },
    },
  },
});

// Dark theme (default)
export const theme: Theme = createTheme(baseTheme, {
  palette: {
    mode: 'dark',
    primary: {
      main: COLORS.KAGGLE_WHITE,
    },
    secondary: {
      main: COLORS.KAGGLE_WHITE,
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    text: {
      primary: COLORS.GREY_200,
      secondary: COLORS.GREY_400,
    },
    divider: COLORS.GREY_800,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        ':root': {
          '--background-color': COLORS.GREY_950,
          '--primary-text-color': COLORS.GREY_200,
          '--secondary-text-color': COLORS.GREY_400,
          '--divider-color': COLORS.GREY_800,
          '--accent-color': COLORS.KAGGLE_FOCUS,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          color: COLORS.KAGGLE_WHITE,
        },
      },
      variants: [
        {
          props: { variant: 'medium' },
          style: {
            backgroundColor: 'transparent',
            color: COLORS.KAGGLE_WHITE,
            border: `1px solid ${COLORS.GREY_400}`,
            '&:hover': {
              backgroundColor: 'transparent',
              border: `1px solid ${COLORS.GREY_500}`,
            },
          },
        },
        {
          props: { variant: 'low' },
          style: {
            backgroundColor: 'transparent',
            color: COLORS.KAGGLE_WHITE,
            border: 'none',
            '&:hover': {
              backgroundColor: COLORS.GREY_800,
            },
          },
        },
      ],
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          color: COLORS.KAGGLE_WHITE,
          '&:hover': {
            backgroundColor: COLORS.GREY_800,
          },
        },
      },
    },
    MuiMenu: {
      styleOverrides: {
        paper: {
          backgroundColor: COLORS.GREY_850,
        },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          '&:hover': {
            backgroundColor: COLORS.GREY_800,
          },
          '&.Mui-focusVisible': {
            backgroundColor: COLORS.GREY_800,
          },
          '&.Mui-selected': {
            backgroundColor: COLORS.GREY_700,
            '&:hover': {
              backgroundColor: COLORS.GREY_700,
            },
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        icon: {
          color: COLORS.GREY_200,
        },
        select: {
          color: COLORS.GREY_200,
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-notchedOutline': {
            border: 'none',
          },
        },
      },
    },
    MuiSlider: {
      styleOverrides: {
        rail: {
          color: COLORS.GREY_600,
          opacity: 1,
        },
        track: {
          color: COLORS.GREY_50,
          opacity: 1,
        },
        valueLabel: {
          backgroundColor: COLORS.KAGGLE_WHITE,
          color: COLORS.KAGGLE_BLACK,
        },
        root: {
          margin: 0,
        },
      },
    },
  },
});

// Light theme variant
export const lightTheme: Theme = createTheme(baseTheme, {
  palette: {
    mode: 'light',
    primary: {
      main: COLORS.KAGGLE_BLACK,
    },
    secondary: {
      main: COLORS.KAGGLE_BLACK,
    },
    background: {
      default: COLORS.KAGGLE_WHITE,
      paper: COLORS.KAGGLE_WHITE,
    },
    text: {
      primary: COLORS.GREY_900,
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        ':root': {
          '--background-color': COLORS.KAGGLE_WHITE,
          '--primary-text-color': COLORS.GREY_900,
          '--secondary-text-color': COLORS.GREY_800,
          '--divider-color': COLORS.GREY_300,
          '--accent-color': COLORS.KAGGLE_FOCUS,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          color: COLORS.GREY_900,
        },
      },
      variants: [
        {
          props: { variant: 'medium' },
          style: {
            backgroundColor: 'transparent',
            color: COLORS.GREY_900,
            border: `1px solid ${COLORS.GREY_400}`,
            '&:hover': {
              backgroundColor: 'transparent',
              border: `1px solid ${COLORS.GREY_900}`,
            },
          },
        },
        {
          props: { variant: 'low' },
          style: {
            backgroundColor: 'transparent',
            color: COLORS.GREY_900,
            border: 'none',
            '&:hover': {
              backgroundColor: COLORS.GREY_100,
            },
          },
        },
      ],
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          color: COLORS.GREY_900,
          '&:hover': {
            backgroundColor: COLORS.GREY_100,
          },
        },
      },
    },
    MuiMenu: {
      styleOverrides: {
        paper: {
          backgroundColor: COLORS.KAGGLE_WHITE,
        },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          '&:hover': {
            backgroundColor: COLORS.GREY_200,
          },
          '&.Mui-focusVisible': {
            backgroundColor: COLORS.GREY_200,
          },
          '&.Mui-selected': {
            backgroundColor: COLORS.GREY_300,
            '&:hover': {
              backgroundColor: COLORS.GREY_300,
            },
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        icon: {
          color: COLORS.GREY_900,
        },
        select: {
          color: COLORS.GREY_900,
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-notchedOutline': {
            border: 'none',
          },
        },
      },
    },
    MuiSlider: {
      styleOverrides: {
        rail: {
          color: COLORS.GREY_600,
        },
        track: {
          color: COLORS.GREY_900,
          opacity: 1,
        },
        valueLabel: {
          backgroundColor: COLORS.GREY_900,
          color: COLORS.KAGGLE_WHITE,
        },
      },
    },
  },
});
