export class VoteVisuals {
  constructor(scene, THREE, votingArcsGroup, targetRingsGroup) {
    this.scene = scene;
    this.THREE = THREE;
    this.votingArcsGroup = votingArcsGroup;
    this.targetRingsGroup = targetRingsGroup;
    
    this.activeVoteArcs = new Map();
    this.activeTargetRings = new Map();
    this.animatingTrails = [];

    // Shared Resources
    this.ringGeometry = new this.THREE.RingGeometry(2, 2.2, 32);
    this.baseRingMaterial = new this.THREE.MeshBasicMaterial({
      color: 0xff0000,
      transparent: true,
      opacity: 0,
      side: this.THREE.DoubleSide,
    });

    // Flat Triangle shape for particles
    const shape = new this.THREE.Shape();
    shape.moveTo(0, 0.6); // Tip (1.5x 0.4)
    shape.lineTo(-0.375, -0.525); // Bottom left (1.5x 0.25, 1.5x 0.35)
    shape.lineTo(0.375, -0.525); // Bottom right
    shape.lineTo(0, 0.6); // Back to tip

    this.particleGeometry = new this.THREE.ShapeGeometry(shape);
    this.particleGeometry.rotateX(Math.PI / 2); // Point tip along +Z (Horizontal)

    this.particleCount = 20;
    this.dummy = new this.THREE.Object3D();
  }

  createVoteTriangleTrail(voter, target, voterName, targetName, color = 0x00ffff) {
    if (!voter || !target) return;

    const startPos = voter.container.position.clone();
    startPos.y += 1.5;
    const endPos = target.container.position.clone();
    endPos.y += 1.5;

    const midPos = new this.THREE.Vector3().addVectors(startPos, endPos).multiplyScalar(0.5);
    const dist = startPos.distanceTo(endPos);
    midPos.y += dist * 0.3;

    const curve = new this.THREE.CatmullRomCurve3([startPos, midPos, endPos]);

    const material = new this.THREE.MeshBasicMaterial({
      color: color,
      transparent: true,
      opacity: 0.8,
      side: this.THREE.DoubleSide, // Ensure visibility from all angles
    });

    const mesh = new this.THREE.InstancedMesh(this.particleGeometry, material, this.particleCount);
    this.votingArcsGroup.add(mesh);

    const trail = {
      mesh,
      curve,
      target: targetName,
      startTime: Date.now(),
      dispose: () => {
        this.votingArcsGroup.remove(mesh);
        mesh.geometry.dispose();
        mesh.material.dispose();
      },
      update: () => {
        const elapsedTime = (Date.now() - trail.startTime) / 1000;
        const speed = 0.5;

        for (let i = 0; i < this.particleCount; i++) {
          const t = (elapsedTime * speed + i / this.particleCount) % 1;

          const pos = curve.getPoint(t);
          const nextT = Math.min(t + 0.01, 1);
          const nextPos = curve.getPoint(nextT);

          this.dummy.position.copy(pos);
          if (t < 0.99) {
            this.dummy.lookAt(nextPos);
          } else {
            const tangent = curve.getTangent(t).normalize();
            this.dummy.lookAt(pos.clone().add(tangent));
          }

          this.dummy.scale.setScalar(0.7);

          this.dummy.updateMatrix();
          mesh.setMatrixAt(i, this.dummy.matrix);
        }
        mesh.instanceMatrix.needsUpdate = true;
      },
    };
    this.activeVoteArcs.set(voterName, trail);
    this.animatingTrails.push(trail);
  }

  updateTargetRing(target, targetName, voteCount, color = 0xff0000) {
    if (!target) return;

    let ringData = this.activeTargetRings.get(targetName);

    if (voteCount > 0 && !ringData) {
      const ring = new this.THREE.Mesh(this.ringGeometry, this.baseRingMaterial.clone());
      ring.position.copy(target.container.position);
      ring.position.y = 0.1;
      ring.rotation.x = -Math.PI / 2;

      this.targetRingsGroup.add(ring);
      ringData = { ring, material: ring.material, targetOpacity: 0 };
      this.activeTargetRings.set(targetName, ringData);
    }

    if (ringData) {
      if (voteCount > 0) {
        ringData.targetOpacity = 0.3 + Math.min(voteCount * 0.2, 0.7);
        ringData.material.color.setHex(color); // Match the action color
      } else {
        ringData.targetOpacity = 0;
      }
    }
  }

  updateVoteVisuals(votes, playerObjects, clearAll = false) {
    if (!playerObjects || playerObjects.size === 0) return;

    if (clearAll) {
      votes.clear();
    }

    this.activeVoteArcs.forEach((trail, voterName) => {
      if (!votes.has(voterName)) {
        trail.dispose();
        this.activeVoteArcs.delete(voterName);
        this.animatingTrails = this.animatingTrails.filter((t) => t !== trail);
      }
    });

    const targetColors = new Map(); // Track color for each target

    votes.forEach((voteData, voterName) => {
      const { target: targetName, type } = voteData;
      const existingTrail = this.activeVoteArcs.get(voterName);

      let color = 0xff0000; // Default to red for better visibility (Day votes)
      if (type === 'night_vote') color = 0xff0000;
      else if (type === 'doctor_heal_action') color = 0x00ff00;
      else if (type === 'seer_inspection') color = 0x0000ff;

      targetColors.set(targetName, color); // Last action color wins for the ring

      if (existingTrail) {
        if (existingTrail.target !== targetName) {
          existingTrail.dispose();
          this.animatingTrails = this.animatingTrails.filter((t) => t !== existingTrail);
          this.createVoteTriangleTrail(playerObjects.get(voterName), playerObjects.get(targetName), voterName, targetName, color);
        }
      } else {
        this.createVoteTriangleTrail(playerObjects.get(voterName), playerObjects.get(targetName), voterName, targetName, color);
      }
    });

    const targetVoteCounts = new Map();
    votes.forEach((voteData) => {
      const { target: targetName } = voteData;
      targetVoteCounts.set(targetName, (targetVoteCounts.get(targetName) || 0) + 1);
    });

    playerObjects.forEach((player, playerName) => {
      this.updateTargetRing(player, playerName, targetVoteCounts.get(playerName) || 0, targetColors.get(playerName));
    });
  }

  update() {
    this.animatingTrails.forEach((trail) => trail.update());

    this.activeTargetRings.forEach((ringData, targetName) => {
      const diff = ringData.targetOpacity - ringData.material.opacity;
      if (Math.abs(diff) > 0.01) {
        ringData.material.opacity += diff * 0.1;
      } else if (ringData.targetOpacity === 0 && ringData.material.opacity > 0) {
        this.targetRingsGroup.remove(ringData.ring);
        ringData.material.dispose(); // Dispose material clone
        this.activeTargetRings.delete(targetName);
      }
    });
  }
}
