---
name: Tactical Surveillance OS
colors:
  surface: '#131314'
  surface-dim: '#131314'
  surface-bright: '#3a393a'
  surface-container-lowest: '#0e0e0f'
  surface-container-low: '#1c1b1c'
  surface-container: '#201f20'
  surface-container-high: '#2a2a2b'
  surface-container-highest: '#353436'
  on-surface: '#e5e2e3'
  on-surface-variant: '#b9ccb2'
  inverse-surface: '#e5e2e3'
  inverse-on-surface: '#313031'
  outline: '#84967e'
  outline-variant: '#3b4b37'
  surface-tint: '#00e639'
  primary: '#ebffe2'
  on-primary: '#003907'
  primary-container: '#00ff41'
  on-primary-container: '#007117'
  inverse-primary: '#006e16'
  secondary: '#ffb3b1'
  on-secondary: '#680011'
  secondary-container: '#ff535a'
  on-secondary-container: '#5b000e'
  tertiary: '#eefbff'
  on-tertiary: '#00363f'
  tertiary-container: '#8ceaff'
  on-tertiary-container: '#006a7a'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#72ff70'
  primary-fixed-dim: '#00e639'
  on-primary-fixed: '#002203'
  on-primary-fixed-variant: '#00530e'
  secondary-fixed: '#ffdad8'
  secondary-fixed-dim: '#ffb3b1'
  on-secondary-fixed: '#410007'
  on-secondary-fixed-variant: '#92001c'
  tertiary-fixed: '#a5eeff'
  tertiary-fixed-dim: '#00daf8'
  on-tertiary-fixed: '#001f25'
  on-tertiary-fixed-variant: '#004e5a'
  background: '#131314'
  on-background: '#e5e2e3'
  surface-variant: '#353436'
typography:
  display-tech:
    fontFamily: Space Grotesk
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-sm:
    fontFamily: Space Grotesk
    fontSize: 18px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: 0.05em
  body-fixed:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.5'
    letterSpacing: 0.01em
  label-mono:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '700'
    lineHeight: '1.0'
    letterSpacing: 0.1em
  telemetry:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: '1.2'
    letterSpacing: 0px
spacing:
  unit: 4px
  gutter: 16px
  margin: 24px
  container-max: 1440px
---

## Brand & Style

This design system is engineered for high-stakes robotics and surveillance environments where precision and speed of information processing are paramount. The brand personality is clinical, authoritative, and advanced, evoking the feeling of a cutting-edge command center. 

The aesthetic style merges **Minimalism** with **Glassmorphism** and technical **Brutalism**. It prioritizes high-contrast data visualization over decorative elements. UI components use razor-sharp edges and microscopic detail to convey a sense of hardware-integrated software. Glassmorphic overlays are used sparingly to maintain context while focusing on specific telemetry data, ensuring the interface feels layered and multi-dimensional without sacrificing professional clarity.

## Colors

The palette is anchored in a "True Black" and "Deep Carbon" foundation to minimize eye strain in low-light command environments and maximize the pop of functional state colors.

- **Primary (Cyberpunk Green):** Reserved exclusively for active scanning, "all-clear" statuses, and system vitality. It should emit a subtle glow effect (10-15px blur) when used in HUD elements.
- **Secondary (Alert Red):** Used for target locks, critical failures, and restricted zones. This color demands immediate attention and overrides other visual information.
- **Tertiary (Data Blue):** Used for auxiliary telemetry, UI guides, and non-critical data readouts.
- **Neutrals:** A range of low-saturation grays provide structure. Borders use a medium-gray to maintain a "wireframe" feel against the black background.

## Typography

This design system utilizes a dual-font approach to balance futuristic character with high legibility. 

**Space Grotesk** is used for headlines and primary data points to provide a technical, geometric edge. **Inter** handles all functional UI, body text, and microscopic telemetry data, ensuring that dense information blocks remain readable at small scales. 

All labels must use uppercase styling with increased letter spacing to mimic military-grade instrumentation. Technical readouts (numbers, coordinates, timestamps) should prioritize monospaced-like tracking within the Inter family to prevent layout shifting during real-time data updates.

## Layout & Spacing

The layout follows a **Rigid Fluid Grid** based on a 4px base unit. Information is packed tightly to maximize the "Data Density" required for surveillance operations. 

- **Grid:** A 12-column system for dashboard layouts, transitioning to a 4-column system for sidebar-heavy robotics controls.
- **Margins:** Consistent 24px outer margins create a "frame" effect, making the screen feel like a dedicated hardware monitor.
- **Modular Boxes:** Content is grouped into "cells" defined by thin 1px borders. These cells should utilize internal padding of 16px (4 units) to maintain internal air while keeping the external footprint compact.
- **Alignment:** Strict adherence to vertical and horizontal axes. Elements should feel "snapped" into place, reinforcing the professional and precise nature of the system.

## Elevation & Depth

Elevation is conveyed through **Tonal Layering** and **Glassmorphism** rather than traditional drop shadows.

1.  **Floor (Level 0):** Pure black (#050505). This is the background for all video feeds and primary maps.
2.  **Plates (Level 1):** Deep Gray (#141416). Used for static sidebars and bottom control docks.
3.  **Overlays (Level 2):** Semi-transparent surfaces (60% opacity) with a 20px backdrop blur. These "Glass" panels are used for floating HUD elements that appear over live video feeds.
4.  **Indicators (Level 3):** High-brightness accents (Green/Red) that appear to "float" above the UI via an outer glow (bloom) effect rather than a shadow, simulating a light-emitting display.

Borders are the primary separators; use 1px solid lines in #2D2D30 for all component boundaries.

## Shapes

The shape language of this design system is **zero-radius**. All containers, buttons, inputs, and icons must feature 90-degree sharp corners. 

This decision reinforces the "Industrial/Military" aesthetic and suggests a no-nonsense, high-precision tool. In rare instances where a circular element is required (e.g., a radar sweep or a camera lens toggle), it must be a perfect geometric circle. Octagonal or "clipped corner" shapes may be used for primary action buttons to add a "futuristic hardware" flair while maintaining the sharp-edge philosophy.

## Components

- **Buttons:** Use 1px outlined borders. The "Searching" state button uses a Cyberpunk Green border and text; the "Alert" state uses Red. Fill should be 5% opacity of the accent color. On hover, the fill increases to 20% and a subtle outer glow is applied to the border.
- **Status Chips:** Small, rectangular tags with no background. Use a leading 4px square "LED" icon that pulses when a process is active.
- **Lists:** Data lists should be separated by thin horizontal rules. Row hovering should trigger a high-contrast background shift to #1F1F22.
- **Checkboxes/Radios:** Square frames only. When selected, the center is filled with a solid primary-colored square (inner 4px margin).
- **Input Fields:** Labeled with "Micro-Labels" in the top-left corner, intersecting the border line. Use a monospaced-style font for entry to ensure coordinate precision.
- **Cards/Cells:** Use "Corner Brackets" (1px L-shaped lines) on the four corners of a card to highlight specific target data within a feed.
- **HUD Crosshairs:** Use thin, 1px lines with "Target Locked" text in Alert Red appearing adjacent to the reticle.