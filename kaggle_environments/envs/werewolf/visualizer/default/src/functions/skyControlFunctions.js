// --- Sky Controls Functions ---
export function createSkyControlsPanel(parent) {
  // Check if panel already exists
  let panel = parent.querySelector('.sky-controls-panel');
  if (panel) return panel;

  panel = document.createElement('div');
  panel.className = 'sky-controls-panel collapsed';
  panel.innerHTML = `
        <div class="sky-controls-header" onclick="toggleSkyControls()">
            <div class="sky-controls-title">🌤️ Sky & Lighting Controls</div>
            <div class="sky-controls-toggle">▼</div>
        </div>
        <div class="sky-controls-content">
            <div class="sky-button-group">
                <button class="sky-button day" onclick="setSkyDayTime()">☀️ Day</button>
                <button class="sky-button night" onclick="setSkyNightTime()">🌙 Night</button>
                <button class="sky-button" onclick="toggleSkyTransition()">🔄 Auto</button>
            </div>
            
            <!-- Sky Parameters -->
            <div class="sky-control-section">
                <div class="sky-section-title">Sky Parameters</div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Turbidity</span>
                        <span class="sky-control-value" id="sky-turbidity-value">10.0</span>
                    </div>
                    <input type="range" id="sky-turbidity" min="0" max="20" step="0.1" value="10" oninput="updateSkyParameter('turbidity', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Rayleigh</span>
                        <span class="sky-control-value" id="sky-rayleigh-value">2.0</span>
                    </div>
                    <input type="range" id="sky-rayleigh" min="0" max="10" step="0.1" value="2" oninput="updateSkyParameter('rayleigh', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Mie Coeff</span>
                        <span class="sky-control-value" id="sky-mie-coeff-value">0.005</span>
                    </div>
                    <input type="range" id="sky-mie-coeff" min="0" max="0.1" step="0.001" value="0.005" oninput="updateSkyParameter('mieCoefficient', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Mie Dir G</span>
                        <span class="sky-control-value" id="sky-mie-g-value">0.8</span>
                    </div>
                    <input type="range" id="sky-mie-g" min="0" max="1" step="0.01" value="0.8" oninput="updateSkyParameter('mieDirectionalG', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Elevation</span>
                        <span class="sky-control-value" id="sky-elevation-value">45°</span>
                    </div>
                    <input type="range" id="sky-elevation" min="-90" max="90" step="1" value="45" oninput="updateSunPosition('elevation', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Azimuth</span>
                        <span class="sky-control-value" id="sky-azimuth-value">180°</span>
                    </div>
                    <input type="range" id="sky-azimuth" min="0" max="360" step="1" value="180" oninput="updateSunPosition('azimuth', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Exposure</span>
                        <span class="sky-control-value" id="sky-exposure-value">0.5</span>
                    </div>
                    <input type="range" id="sky-exposure" min="0" max="2" step="0.01" value="1.0" oninput="updateExposure(this.value)">
                </div>
            </div>
            
            <!-- Lighting -->
            <div class="sky-control-section">
                <div class="sky-section-title">Lighting</div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Sun Intensity</span>
                        <span class="sky-control-value" id="sky-sun-intensity-value">1.0</span>
                    </div>
                    <input type="range" id="sky-sun-intensity" min="0" max="3" step="0.1" value="1" oninput="updateLighting('sunIntensity', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Moon Intensity</span>
                        <span class="sky-control-value" id="sky-moon-intensity-value">0.5</span>
                    </div>
                    <input type="range" id="sky-moon-intensity" min="0" max="1" step="0.01" value="0.5" oninput="updateLighting('moonIntensity', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Ambient</span>
                        <span class="sky-control-value" id="sky-ambient-value">0.4</span>
                    </div>
                    <input type="range" id="sky-ambient" min="0" max="1" step="0.01" value="0.1" oninput="updateLighting('ambientIntensity', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Time of Day</span>
                        <span class="sky-control-value" id="sky-time-value">12:00</span>
                    </div>
                    <input type="range" id="sky-time-slider" min="0" max="24" step="0.1" value="12" oninput="setTimeOfDay(this.value)">
                </div>
            </div>
            
            <!-- Post-Processing -->
            <div class="sky-control-section">
                <div class="sky-section-title">Bloom</div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Strength</span>
                        <span class="sky-control-value" id="sky-bloom-strength-value">0.6</span>
                    </div>
                    <input type="range" id="sky-bloom-strength" min="0" max="1" step="0.01" value="0.6" oninput="updateBloom('strength', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Radius</span>
                        <span class="sky-control-value" id="sky-bloom-radius-value">2.0</span>
                    </div>
                    <input type="range" id="sky-bloom-radius" min="0" max="2" step="0.01" value="2.0" oninput="updateBloom('radius', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Threshold</span>
                        <span class="sky-control-value" id="sky-bloom-threshold-value">0.1</span>
                    </div>
                    <input type="range" id="sky-bloom-threshold" min="0" max="1" step="0.01" value="0.1" oninput="updateBloom('threshold', this.value)">
                </div>
            </div>
            
            <!-- Clouds -->
            <div class="sky-control-section">
                <div class="sky-section-title">Clouds</div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Opacity</span>
                        <span class="sky-control-value" id="sky-cloud-opacity-value">0.5</span>
                    </div>
                    <input type="range" id="sky-cloud-opacity" min="0" max="1" step="0.01" value="0.5" oninput="updateClouds('opacity', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Speed</span>
                        <span class="sky-control-value" id="sky-cloud-speed-value">1.0</span>
                    </div>
                    <input type="range" id="sky-cloud-speed" min="0" max="5" step="0.1" value="1" oninput="updateClouds('speed', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Enable</span>
                    </div>
                    <input type="checkbox" id="sky-cloud-toggle" checked onchange="toggleClouds(this.checked)">
                </div>
            </div>
            
            <!-- God Rays -->
            <div class="sky-control-section">
                <div class="sky-section-title">God Rays</div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Intensity</span>
                        <span class="sky-control-value" id="sky-godray-intensity-value">2.0</span>
                    </div>
                    <input type="range" id="sky-godray-intensity" min="0" max="2" step="0.1" value="2.0" oninput="updateGodRays('intensity', this.value)">
                </div>
                
                <div class="sky-control-item">
                    <div class="sky-control-label">
                        <span>Enable</span>
                    </div>
                    <input type="checkbox" id="sky-godray-toggle" checked onchange="toggleGodRays(this.checked)">
                </div>
            </div>
            
            <!-- Status -->
            <div class="sky-control-section">
                <div class="sky-section-title">Status</div>
                <div class="sky-info-panel">
                    <div class="sky-info-item">
                        <span class="sky-info-label">Phase:</span>
                        <span class="sky-info-value"><span id="sky-current-phase" class="sky-phase-indicator sky-phase-day">DAY</span></span>
                    </div>
                    <div class="sky-info-item">
                        <span class="sky-info-label">Sun:</span>
                        <span class="sky-info-value" id="sky-sun-position">Az: 180°, El: 45°</span>
                    </div>
                    <div class="sky-info-item">
                        <span class="sky-info-label">Moon:</span>
                        <span class="sky-info-value" id="sky-moon-position">Hidden</span>
                    </div>
                </div>
            </div>
        </div>
    `;

  parent.appendChild(panel);
  return panel;
}

export function toggleSkyControls() {
  const panel = document.querySelector('.sky-controls-panel');
  if (panel) {
    panel.classList.toggle('collapsed');
  }
}

export function updateSkyParameter(param, value) {
  const floatValue = parseFloat(value);

  // Update display
  let displayValue = floatValue.toFixed(param === 'mieCoefficient' ? 3 : 1);
  const valueId =
    param === 'mieCoefficient'
      ? 'sky-mie-coeff-value'
      : param === 'mieDirectionalG'
        ? 'sky-mie-g-value'
        : `sky-${param}-value`;
  const valueEl = document.getElementById(valueId);
  if (valueEl) valueEl.textContent = displayValue;

  if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._sky) {
    const sky = window.werewolfThreeJs.demo._sky;
    if (sky.material && sky.material.uniforms && sky.material.uniforms[param]) {
      sky.material.uniforms[param].value = floatValue;
      console.debug(`[Sky Controls] Updated ${param} to ${displayValue}`);
    }
  }
}

export function updateSunPosition(type, value) {
  const floatValue = parseFloat(value);
  const valueEl = document.getElementById(`sky-${type}-value`);
  if (valueEl) valueEl.textContent = `${floatValue}°`;

  if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._sky) {
    const sky = window.werewolfThreeJs.demo._sky;

    // Get current values
    const elevationEl = document.getElementById('sky-elevation');
    const azimuthEl = document.getElementById('sky-azimuth');
    const currentElevation = elevationEl ? parseFloat(elevationEl.value) : 45;
    const currentAzimuth = azimuthEl ? parseFloat(azimuthEl.value) : 180;

    // Convert to radians
    const phi = ((90 - currentElevation) * Math.PI) / 180;
    const theta = (currentAzimuth * Math.PI) / 180;

    // Calculate sun position
    const sunX = Math.sin(phi) * Math.cos(theta);
    const sunY = Math.cos(phi);
    const sunZ = Math.sin(phi) * Math.sin(theta);

    if (sky.material && sky.material.uniforms && sky.material.uniforms['sunPosition']) {
      sky.material.uniforms['sunPosition'].value.set(sunX, sunY, sunZ);
    }

    // Update lighting
    updateDayNightLighting(currentElevation);
    updateSkyInfo();

    console.debug(`[Sky Controls] Sun position - Elevation: ${currentElevation}°, Azimuth: ${currentAzimuth}°`);
  }
}

export function updateExposure(value) {
  const floatValue = parseFloat(value);
  const valueEl = document.getElementById('sky-exposure-value');
  if (valueEl) valueEl.textContent = floatValue.toFixed(2);

  if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._renderer) {
    window.werewolfThreeJs.demo._renderer.toneMappingExposure = floatValue;
    console.debug(`[Sky Controls] Updated exposure to ${floatValue.toFixed(2)}`);
  }
}

export function updateLighting(type, value) {
  const floatValue = parseFloat(value);
  const valueId =
    type === 'sunIntensity'
      ? 'sky-sun-intensity-value'
      : type === 'moonIntensity'
        ? 'sky-moon-intensity-value'
        : 'sky-ambient-value';
  const valueEl = document.getElementById(valueId);
  if (valueEl) valueEl.textContent = floatValue.toFixed(1);

  if (window.werewolfThreeJs && window.werewolfThreeJs.demo) {
    const demo = window.werewolfThreeJs.demo;

    switch (type) {
      case 'sunIntensity':
        if (demo._sunLight) {
          demo._sunLight.intensity = floatValue;
        }
        break;
      case 'moonIntensity':
        if (demo._moonLight) {
          demo._moonLight.intensity = floatValue;
        }
        break;
      case 'ambientIntensity':
        if (demo._ambientLight) {
          demo._ambientLight.intensity = floatValue;
        }
        break;
    }

    console.debug(`[Sky Controls] Updated ${type} to ${floatValue.toFixed(1)}`);
  }
}

export function updateBloom(type, value) {
  const floatValue = parseFloat(value);
  const valueId = `sky-bloom-${type}-value`;
  const valueEl = document.getElementById(valueId);
  if (valueEl) valueEl.textContent = floatValue.toFixed(2);

  if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._bloomPass) {
    const bloomPass = window.werewolfThreeJs.demo._bloomPass;

    switch (type) {
      case 'strength':
        bloomPass.strength = floatValue;
        break;
      case 'radius':
        bloomPass.radius = floatValue;
        break;
      case 'threshold':
        bloomPass.threshold = floatValue;
        break;
    }

    console.debug(`[Sky Controls] Updated bloom ${type} to ${floatValue.toFixed(2)}`);
  }
}

export function updateClouds(type, value) {
  const floatValue = parseFloat(value);
  const valueId = `sky-cloud-${type}-value`;
  const valueEl = document.getElementById(valueId);
  if (valueEl) valueEl.textContent = floatValue.toFixed(1);

  if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._clouds) {
    const clouds = window.werewolfThreeJs.demo._clouds;

    clouds.forEach((cloud) => {
      if (!cloud || !cloud.material) return;

      switch (type) {
        case 'opacity':
          cloud.material.opacity = floatValue;
          break;
        case 'speed':
          cloud.userData = cloud.userData || {};
          cloud.userData.speed = floatValue * 0.0001;
          break;
      }
    });

    console.debug(`[Sky Controls] Updated cloud ${type} to ${floatValue.toFixed(1)}`);
  }
}

export function toggleClouds(enabled) {
  if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._clouds) {
    window.werewolfThreeJs.demo._clouds.forEach((cloud) => {
      if (cloud) cloud.visible = enabled;
    });
    console.debug(`[Sky Controls] Clouds ${enabled ? 'enabled' : 'disabled'}`);
  }
}

export function updateGodRays(type, value) {
  const floatValue = parseFloat(value);
  const valueEl = document.getElementById('sky-godray-intensity-value');
  if (valueEl) valueEl.textContent = floatValue.toFixed(1);

  if (window.werewolfThreeJs && window.werewolfThreeJs.demo) {
    const demo = window.werewolfThreeJs.demo;

    if (type === 'intensity') {
      if (demo.setGodRayIntensity) {
        demo.setGodRayIntensity(floatValue);
      } else if (demo._godRayIntensity !== undefined) {
        demo._godRayIntensity = floatValue;
      }
    }

    console.debug(`[Sky Controls] Updated god ray ${type} to ${floatValue.toFixed(1)}`);
  }
}

export function toggleGodRays(enabled) {
  if (window.werewolfThreeJs && window.werewolfThreeJs.demo) {
    const demo = window.werewolfThreeJs.demo;
    const intensityEl = document.getElementById('sky-godray-intensity');
    const intensity = intensityEl ? parseFloat(intensityEl.value) : 1.0;

    if (enabled) {
      demo._godRayIntensity = intensity;
    } else {
      demo._godRayIntensity = 0;
    }
    console.debug(`[Sky Controls] God rays ${enabled ? 'enabled' : 'disabled'}`);
  }
}

export function updateDayNightLighting(elevation) {
  if (window.werewolfThreeJs && window.werewolfThreeJs.demo) {
    const demo = window.werewolfThreeJs.demo;
    const isDay = elevation > 0;

    // Update phase indicator
    const phaseIndicator = document.getElementById('sky-current-phase');
    if (phaseIndicator) {
      phaseIndicator.textContent = isDay ? 'DAY' : 'NIGHT';
      phaseIndicator.className = `sky-phase-indicator ${isDay ? 'sky-phase-day' : 'sky-phase-night'}`;
    }

    // Trigger phase update if available
    if (demo.updateSkyForPhase) {
      demo.updateSkyForPhase(isDay);
    }
  }
}

export function setSkyDayTime() {
  console.debug('[Sky Controls] Setting day time');
  const elevationEl = document.getElementById('sky-elevation');
  const azimuthEl = document.getElementById('sky-azimuth');
  const timeEl = document.getElementById('sky-time-slider');

  if (elevationEl) elevationEl.value = 45;
  if (azimuthEl) azimuthEl.value = 180;
  if (timeEl) timeEl.value = 12;

  updateSunPosition('elevation', 45);
  updateSunPosition('azimuth', 180);
  setTimeOfDay(12);
}

export function setSkyNightTime() {
  console.debug('[Sky Controls] Setting night time');
  const elevationEl = document.getElementById('sky-elevation');
  const azimuthEl = document.getElementById('sky-azimuth');
  const timeEl = document.getElementById('sky-time-slider');

  if (elevationEl) elevationEl.value = -45;
  if (azimuthEl) azimuthEl.value = 0;
  if (timeEl) timeEl.value = 0;

  updateSunPosition('elevation', -45);
  updateSunPosition('azimuth', 0);
  setTimeOfDay(0);
}

export function setTimeOfDay(value) {
  const hours = parseFloat(value);
  const h = Math.floor(hours);
  const m = Math.floor((hours - h) * 60);

  const timeValueEl = document.getElementById('sky-time-value');
  if (timeValueEl) {
    timeValueEl.textContent = `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}`;
  }

  if (window.werewolfThreeJs && window.werewolfThreeJs.demo && window.werewolfThreeJs.demo._sky) {
    // Convert hours to sun position
    const sunAngle = (hours / 24) * Math.PI * 2 - Math.PI / 2;
    const elevation = Math.sin(sunAngle) * 90;
    const azimuth = (hours / 24) * 360;

    // Update sliders
    const elevationEl = document.getElementById('sky-elevation');
    const azimuthEl = document.getElementById('sky-azimuth');
    if (elevationEl) elevationEl.value = elevation;
    if (azimuthEl) azimuthEl.value = azimuth;

    // Update sun position
    updateSunPosition('elevation', elevation);
    updateSunPosition('azimuth', azimuth);

    console.debug(`[Sky Controls] Time updated to ${h}:${m.toString().padStart(2, '0')}`);
  }

  updateSkyInfo();
}

let skyTransitionInterval = null;

export function toggleSkyTransition() {
  if (skyTransitionInterval) {
    clearInterval(skyTransitionInterval);
    skyTransitionInterval = null;
    console.debug('[Sky Controls] Auto transition stopped');
  } else {
    console.debug('[Sky Controls] Auto transition started');
    skyTransitionInterval = setInterval(() => {
      const timeEl = document.getElementById('sky-time-slider');
      if (timeEl) {
        let currentTime = parseFloat(timeEl.value);
        currentTime += 0.1;
        if (currentTime >= 24) currentTime = 0;
        timeEl.value = currentTime;
        setTimeOfDay(currentTime);
      }
    }, 100);
  }
}

export function updateSkyInfo() {
  if (!window.werewolfThreeJs || !window.werewolfThreeJs.demo) return;

  const demo = window.werewolfThreeJs.demo;

  if (demo._sky && demo._sky.material && demo._sky.material.uniforms) {
    const sunPos = demo._sky.material.uniforms['sunPosition'].value;

    // Calculate sun angles
    const elevation = Math.asin(sunPos.y) * (180 / Math.PI);
    const azimuth = Math.atan2(sunPos.x, sunPos.z) * (180 / Math.PI);

    const sunPosEl = document.getElementById('sky-sun-position');
    if (sunPosEl) {
      sunPosEl.textContent = `Az: ${azimuth.toFixed(1)}°, El: ${elevation.toFixed(1)}°`;
    }

    // Moon position (opposite of sun)
    const moonPosEl = document.getElementById('sky-moon-position');
    if (moonPosEl) {
      if (elevation < 0) {
        const moonElevation = -elevation;
        const moonAzimuth = (azimuth + 180) % 360;
        moonPosEl.textContent = `Az: ${moonAzimuth.toFixed(1)}°, El: ${moonElevation.toFixed(1)}°`;
      } else {
        moonPosEl.textContent = 'Hidden';
      }
    }
  }
}