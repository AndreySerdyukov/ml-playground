# ML Playground – ml-design assets

```
ml-design/
  logo/mark.svg          logo mark (24x24)
  logo/lockup.svg        mark + "ML Playground" wordmark (200x24)
  icons/*.svg            9 icons, 24x24 grid
  ModelIcon.jsx          PATHS object + React component
```

Model icons: wine, diamonds, cars, bayesian, loans, uplift.
Feature icons: model-agnostic-forms, instant-prediction, notebook-to-browser.

Style: monochrome line, fill="none", stroke="currentColor", stroke-width 1.5,
round caps/joins, 24x24 with ~2px padding. Color comes from CSS `color`,
so the same file works on #ffffff and on #1d1d1f.

Usage:
```jsx
import { ModelIcon } from './ml-design/ModelIcon';
<ModelIcon name="uplift" size={32} style={{ color: '#1d1d1f' }} />
```

The full specimen sheet lives in `ML Playground Icon System.dc.html`
(also exported as `ml-design/specimen.html`).
