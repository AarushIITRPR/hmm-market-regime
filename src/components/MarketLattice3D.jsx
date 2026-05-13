import { useEffect, useRef } from "react";
import * as THREE from "three";

export default function MarketLattice3D() {
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) {
      return undefined;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.1, 120);
    camera.position.set(0, 5.4, 21);

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    const grid = new THREE.GridHelper(42, 42, 0x19f5c8, 0x153d4a);
    grid.position.y = -4;
    grid.material.transparent = true;
    grid.material.opacity = 0.32;
    group.add(grid);

    const pointCount = window.innerWidth < 720 ? 70 : 130;
    const points = new Float32Array(pointCount * 3);
    const colors = new Float32Array(pointCount * 3);
    const palette = [
      new THREE.Color("#19f5c8"),
      new THREE.Color("#6ee7ff"),
      new THREE.Color("#f7c948"),
      new THREE.Color("#ff5c7a"),
    ];

    for (let index = 0; index < pointCount; index += 1) {
      const offset = index * 3;
      const x = (Math.random() - 0.5) * 34;
      const z = (Math.random() - 0.5) * 26;
      const y = Math.sin(index * 0.35) * 2.2 + (Math.random() - 0.5) * 1.4;
      points[offset] = x;
      points[offset + 1] = y;
      points[offset + 2] = z;

      const color = palette[index % palette.length];
      colors[offset] = color.r;
      colors[offset + 1] = color.g;
      colors[offset + 2] = color.b;
    }

    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute("position", new THREE.BufferAttribute(points, 3));
    pointGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const pointMaterial = new THREE.PointsMaterial({
      size: 0.12,
      vertexColors: true,
      transparent: true,
      opacity: 0.82,
    });
    const pointMesh = new THREE.Points(pointGeometry, pointMaterial);
    group.add(pointMesh);

    const linePositions = [];
    for (let index = 0; index < pointCount - 1; index += 2) {
      const a = index * 3;
      const b = (index + 1) * 3;
      linePositions.push(
        points[a],
        points[a + 1],
        points[a + 2],
        points[b],
        points[b + 1],
        points[b + 2],
      );
    }
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
    const lineMaterial = new THREE.LineBasicMaterial({
      color: "#19f5c8",
      transparent: true,
      opacity: 0.24,
    });
    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    group.add(lines);

    const barCount = 28;
    const barGeometry = new THREE.BoxGeometry(0.18, 1, 0.18);
    const barMaterial = new THREE.MeshBasicMaterial({
      color: "#6ee7ff",
      transparent: true,
      opacity: 0.68,
    });
    const bars = new THREE.InstancedMesh(barGeometry, barMaterial, barCount);
    const dummy = new THREE.Object3D();
    for (let index = 0; index < barCount; index += 1) {
      const height = 0.8 + Math.abs(Math.sin(index * 0.72)) * 5;
      dummy.position.set((index - barCount / 2) * 0.72, -4 + height / 2, -8.8);
      dummy.scale.set(1, height, 1);
      dummy.updateMatrix();
      bars.setMatrixAt(index, dummy.matrix);
    }
    group.add(bars);

    const coreGeometry = new THREE.IcosahedronGeometry(3.2, 1);
    const coreMaterial = new THREE.MeshBasicMaterial({
      color: "#19f5c8",
      wireframe: true,
      transparent: true,
      opacity: 0.26,
    });
    const core = new THREE.Mesh(coreGeometry, coreMaterial);
    core.position.set(window.innerWidth < 760 ? 4.8 : 10.2, 2.1, -6.5);
    group.add(core);

    const ringGeometry = new THREE.TorusGeometry(4.5, 0.012, 8, 96);
    const ringMaterial = new THREE.MeshBasicMaterial({
      color: "#6ee7ff",
      transparent: true,
      opacity: 0.28,
    });
    const rings = new THREE.Group();
    for (let index = 0; index < 3; index += 1) {
      const ring = new THREE.Mesh(ringGeometry, ringMaterial);
      ring.rotation.x = Math.PI / 2 + index * 0.52;
      ring.rotation.y = index * 0.74;
      rings.add(ring);
    }
    rings.position.copy(core.position);
    group.add(rings);

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", handleResize);

    let frameId = 0;
    const clock = new THREE.Clock();
    const animate = () => {
      const elapsed = clock.getElapsedTime();
      group.rotation.y = Math.sin(elapsed * 0.12) * 0.18;
      group.rotation.x = -0.12 + Math.sin(elapsed * 0.08) * 0.03;
      pointMesh.rotation.y = elapsed * 0.045;
      lines.rotation.y = elapsed * 0.045;
      bars.position.x = Math.sin(elapsed * 0.45) * 0.35;
      core.rotation.x = elapsed * 0.16;
      core.rotation.y = elapsed * 0.21;
      rings.rotation.y = elapsed * 0.1;
      rings.rotation.z = Math.sin(elapsed * 0.2) * 0.2;
      renderer.render(scene, camera);
      frameId = window.requestAnimationFrame(animate);
    };
    animate();

    return () => {
      window.cancelAnimationFrame(frameId);
      window.removeEventListener("resize", handleResize);
      pointGeometry.dispose();
      pointMaterial.dispose();
      lineGeometry.dispose();
      lineMaterial.dispose();
      barGeometry.dispose();
      barMaterial.dispose();
      coreGeometry.dispose();
      coreMaterial.dispose();
      ringGeometry.dispose();
      ringMaterial.dispose();
      grid.geometry.dispose();
      grid.material.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  return <div className="market-lattice" ref={mountRef} aria-hidden="true" />;
}
