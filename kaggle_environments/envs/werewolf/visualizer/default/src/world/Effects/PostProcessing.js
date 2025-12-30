export class PostProcessing {
  constructor(scene, camera, renderer, width, height, THREE, EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, FilmPass) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.width = width;
    this.height = height;
    this.THREE = THREE;
    
    // Modules
    this.EffectComposer = EffectComposer;
    this.RenderPass = RenderPass;
    this.UnrealBloomPass = UnrealBloomPass;
    this.ShaderPass = ShaderPass;
    this.FilmPass = FilmPass;

    this.composer = null;
    this.bloomPass = null;
    this.filmPass = null;
    this.atmospherePass = null;

    this.init();
  }

  init() {
    this.composer = new this.EffectComposer(this.renderer);
    
    // Optimize: Cap internal render resolution to 1.0 to save massive memory on High-DPI screens
    // The canvas is scaled by CSS, so it still looks okay, just internal buffers are smaller.
    const pixelRatio = 1.0; 
    this.composer.setSize(this.width * pixelRatio, this.height * pixelRatio);

    const renderPass = new this.RenderPass(this.scene, this.camera);
    this.composer.addPass(renderPass);

    // Optimize Bloom: Use quarter resolution of the ALREADY reduced composer resolution
    const bloomResolution = new this.THREE.Vector2(this.width * pixelRatio * 0.25, this.height * pixelRatio * 0.25);
    this.bloomPass = new this.UnrealBloomPass(
      bloomResolution,
      0.15,
      0.4,
      0.85
    );
    this.composer.addPass(this.bloomPass);

    const atmosphereShader = {
      uniforms: {
        'tDiffuse': { value: null },
        'uTint': { value: new this.THREE.Vector3(1, 1, 1) },
        'uContrast': { value: 1.0 },
        'uVignetteBase': { value: 0.7 },
      },
      vertexShader: `
            varying vec2 vUv;
            void main() {
              vUv = uv;
              gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
          `,
      fragmentShader: `
            uniform sampler2D tDiffuse;
            uniform vec3 uTint;
            uniform float uContrast;
            uniform float uVignetteBase;
            varying vec2 vUv;
            
            void main() {
              vec4 color = texture2D(tDiffuse, vUv);

              // 1. Tint
              color.rgb *= uTint;

              // 2. Contrast (Approximate with simple power)
              // optimization: only apply if uContrast != 1.0? No, branching is bad.
              // Just do it, usage is low.
              color.rgb = pow(color.rgb, vec3(uContrast));

              // 3. Vignette (Squared distance for speed)
              vec2 d = vUv - 0.5;
              float distSq = dot(d, d); // 0 at center, 0.5 at corners (0.707^2)
              // Smoothstep equivalent: smoothstep(0.4^2, 1.0^2, distSq) ~ range 0.16 to 1.0
              // actually dist is 0 to 0.707. 0.4 radius is 0.16 sq.

              float vignette = 1.0 - smoothstep(0.16, 0.5, distSq);
              color.rgb *= mix(uVignetteBase, 1.0, vignette);

              gl_FragColor = color;
            }
          `,
    };

    this.atmospherePass = new this.ShaderPass(atmosphereShader);
    this.composer.addPass(this.atmospherePass);
  }

  update(time, phaseValue) {
    if (this.bloomPass) {
      this.bloomPass.strength = 0.1 + phaseValue * 0.03;
      this.bloomPass.radius = 0.08 + phaseValue * 0.04;
      this.bloomPass.threshold = 0.0 + phaseValue * 0.1;
    }

    if (this.atmospherePass) {
      // Calculate visuals on CPU to save GPU cycles
      const uTint = this.atmospherePass.uniforms.uTint.value;
      const uniforms = this.atmospherePass.uniforms;

      if (phaseValue > 0.49) {
        // Night Phase
        // Target: slightly red/purple, dark
        // Tint: mix(white, nightTint, 0.4) * 0.9 from original shader
        // NightTint was (1.1, 0.8, 0.9)
        // Enhanced Red was +0.2 * r -- handled by just boosting R in tint?
        // Let's approximate the final composite tint:
        uTint.set(1.15, 0.85, 0.95); // Night Tint
        uniforms.uContrast.value = 1.1;
        uniforms.uVignetteBase.value = 0.7; // Dark borders
      } else {
        // Day Phase
        // Target: mix(original, warm, 0.2)
        uTint.set(1.02, 1.0, 0.98); // Subtle Warmth
        uniforms.uContrast.value = 1.0;
        uniforms.uVignetteBase.value = 0.85; // Lighter borders
      }
    }
  }

  render() {
    this.composer.render();
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    
    if (this.composer) {
        // Optimize: Keep resolution capped at 1.0
        const pixelRatio = 1.0; 
        this.composer.setSize(width * pixelRatio, height * pixelRatio);
    }
    
    if (this.bloomPass) {
        // Bloom resolution relative to the composer size
      this.bloomPass.resolution.set(width * 0.25, height * 0.25);
    }
  }
}
