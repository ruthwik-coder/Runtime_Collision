---
name: Kinetic Response System
colors:
  surface: '#0f1418'
  surface-dim: '#0f1418'
  surface-bright: '#343a3e'
  surface-container-lowest: '#0a0f12'
  surface-container-low: '#171c20'
  surface-container: '#1b2024'
  surface-container-high: '#252b2e'
  surface-container-highest: '#303539'
  on-surface: '#dee3e8'
  on-surface-variant: '#bdc8d1'
  inverse-surface: '#dee3e8'
  inverse-on-surface: '#2c3135'
  outline: '#87929a'
  outline-variant: '#3e484f'
  surface-tint: '#7bd0ff'
  primary: '#8ed5ff'
  on-primary: '#00354a'
  primary-container: '#38bdf8'
  on-primary-container: '#004965'
  inverse-primary: '#00668a'
  secondary: '#b9c8de'
  on-secondary: '#233143'
  secondary-container: '#39485a'
  on-secondary-container: '#a7b6cc'
  tertiary: '#ffc176'
  on-tertiary: '#472a00'
  tertiary-container: '#f1a02b'
  on-tertiary-container: '#613b00'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#c4e7ff'
  primary-fixed-dim: '#7bd0ff'
  on-primary-fixed: '#001e2c'
  on-primary-fixed-variant: '#004c69'
  secondary-fixed: '#d4e4fa'
  secondary-fixed-dim: '#b9c8de'
  on-secondary-fixed: '#0d1c2d'
  on-secondary-fixed-variant: '#39485a'
  tertiary-fixed: '#ffddb8'
  tertiary-fixed-dim: '#ffb960'
  on-tertiary-fixed: '#2a1700'
  on-tertiary-fixed-variant: '#653e00'
  background: '#0f1418'
  on-background: '#dee3e8'
  surface-variant: '#303539'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 30px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-base:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  data-lg:
    fontFamily: JetBrains Mono
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
  data-sm:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '700'
    lineHeight: 12px
    letterSpacing: 0.05em
spacing:
  grid-unit: 4px
  container-padding: 16px
  element-gap: 8px
  section-margin: 24px
  sidebar-width: 280px
---

## Brand & Style
The design system is engineered for high-stakes, real-time monitoring environments where cognitive load management is critical. The brand personality is authoritative, precise, and utilitarian, mirroring the aesthetics of industrial control systems and aerospace telemetry dashboards. 

The visual style employs a **Technical Minimalist** approach with a heavy emphasis on **Border-Defined Hierarchy**. Rather than using shadows or soft blurs, the system relies on sharp lines, distinct color coding, and high-density data visualization to communicate status instantly. The interface should feel like a high-performance tool—cold, efficient, and reliable—minimizing decorative elements in favor of functional clarity.

## Colors
This design system utilizes a deep, nocturnal foundation to maximize the visibility of status-critical colors. The palette is strictly functional:

- **Base Foundation**: The core background is `#0F172A`, providing a low-glare surface for long-duration monitoring.
- **Surface & Borders**: UI sections are defined by `#1E293B` surfaces and crisp `#334155` borders.
- **Primary Action**: A technical blue (`#38BDF8`) is used for active states and interactive highlights.
- **Status Indicators**: High-saturation Red, Amber, and Green are reserved exclusively for situational awareness. Red indicates a confirmed accident, Amber signals a traffic anomaly or sensor fault, and Green confirms active system health.

## Typography
The typography system distinguishes between **UI Controls** (Inter) and **Operational Data** (JetBrains Mono).

- **Inter** is used for all navigational elements, modal headers, and descriptive text to ensure readability and professional tone.
- **JetBrains Mono** is employed for all telemetry, timestamps, coordinates, and sensor readings. The monospaced nature ensures that fluctuating numerical data remains vertically aligned in tables and grids, preventing visual "jumping" during real-time updates.
- All labels for data points should be set in `label-caps` to provide a clear distinction between the "key" and the "value."

## Layout & Spacing
The layout follows a **Rigid Grid** model designed for high density. The interface is divided into functional "Control Blocks" using a consistent 4px base unit.

- **Dashboard Layout**: A fixed sidebar for primary navigation and global alerts, with a multi-pane main stage.
- **Panes**: Content is grouped into bordered panels. These panels should use `element-gap` (8px) for internal items to maximize the amount of information visible on a single screen without scrolling.
- **Alignment**: All data points must be top-left aligned within their respective cells. In data-heavy tables, use a 1px border separator rather than whitespace to maintain the industrial, structured feel.

## Elevation & Depth
In this design system, depth is communicated through **Tonal Layering** and **Luminescence** rather than shadows.

- **Level 0 (Floor)**: The main background (`#0F172A`).
- **Level 1 (Panels)**: Surface areas (`#1E293B`) with a 1px solid border (`#334155`).
- **Level 2 (Interaction/Popovers)**: Elements that sit above the main plane use a slightly lighter border (`#475569`) and a subtle 10% opacity tint of the primary blue to indicate focus.
- **Active States**: Use "Glow" effects (inner shadows or drop shadows with 0 blur and 2px spread) only for critical alerts to simulate physical LED indicators.

## Shapes
To reinforce the industrial and military aesthetic, the design system utilizes **Sharp (0px)** roundedness. Every button, input, panel, and indicator features hard 90-degree corners. This evokes a sense of precision and maximizes the available pixel space for data display, which is critical in high-density environments.

## Components
### Buttons & Controls
- **Primary**: Solid `#38BDF8` with black text. Sharp corners.
- **Secondary**: Outlined with `#334155` and white text. 
- **Status Buttons**: For emergency actions (e.g., "Dispatch"), use a full-width Danger Red button with white text and a 1px white inner border.

### Data Chips
- Small, rectangular indicators with a background tint (20% opacity of status color) and a solid 1px border of the status color. Text must be JetBrains Mono.

### Input Fields
- Dark background (`#020617`), 1px border. On focus, the border changes to the primary blue. Placeholder text should be `secondary-color` at 50% opacity.

### Monitoring Cards
- Headers should have a subtle top-border (2px) in the color corresponding to the current status of that highway segment.

### Tables
- Header cells should have a background of `#334155` with uppercase monospaced labels. Row heights should be condensed (32px) to allow for 20+ rows on a standard view.

### Tactical Map
- Use a monochromatic dark map style. Accidents are marked with pulsing red diamond icons; sensors are marked with small white or green dots.