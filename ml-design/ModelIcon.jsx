export const PATHS = {
  logo: [
    'M6.5 3.5h11a3 3 0 0 1 3 3v11a3 3 0 0 1-3 3h-11a3 3 0 0 1-3-3v-11a3 3 0 0 1 3-3z',
    'M10 8.5l6 3.5-6 3.5z',
  ],
  wine: [
    'M8.5 3.5h7v4.5a3.5 3.5 0 0 1-7 0z',
    'M8.5 6.6c1.2 1 2.3 1 3.5 0s2.3-1 3.5 0',
    'M12 11.5v7',
    'M9.5 18.5h5',
  ],
  diamonds: [
    'M8.5 4.5h7l4 3-7.5 12-7.5-12z',
    'M4.5 7.5h15',
    'M8.5 7.5L12 19.5 15.5 7.5',
  ],
  cars: [
    'M3.5 15.2c1.6-3.2 4.2-5.2 7.6-5.2 4.6 0 8.4 2.8 9.9 5.6',
    'M7.4 13.6c1.8-1 3.8-1.3 5.8-1.1',
    'M10.6 16.4h3',
    'M5.6 16.4a2.6 2.6 0 0 1 5 0',
    'M13.6 16.4a2.6 2.6 0 0 1 5 0',
  ],
  bayesian: [
    'M3 18h18',
    'M3.5 18c4 0 3.5-11 8.5-11s4.5 11 8.5 11',
    'M12 7.5v10.5',
  ],
  loans: [
    'M5.5 7h13a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2z',
    'M12 9.2v5.9',
    'M14.1 11c0-1-.9-1.6-2.1-1.6s-2.1.6-2.1 1.5.8 1.3 2.1 1.6 2.1.8 2.1 1.7-.9 1.5-2.1 1.5-2.1-.6-2.1-1.5',
  ],
  uplift: [
    'M3.5 18.5h17',
    'M4 15.5l4.5-3.5 3.5 2.5 6.5-6',
    'M14 8.5h4.5v4.5',
  ],
  'model-agnostic-forms': [
    'M4.5 3.5h15a1 1 0 0 1 1 1v15a1 1 0 0 1-1 1h-15a1 1 0 0 1-1-1v-15a1 1 0 0 1 1-1z',
    'M7.5 8.5h9',
    'M7.5 12h9',
    'M7.5 15.5h5',
  ],
  'instant-prediction': [
    'M13.5 3l-7 11h5l-1 7 7-11h-5z',
  ],
  'notebook-to-browser': [
    'M4.5 4.5h15a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-15a1 1 0 0 1-1-1v-13a1 1 0 0 1 1-1z',
    'M3.5 8.5h17',
    'M6.4 6.5h.2',
    'M8.9 6.5h.2',
    'M12 10.5v5',
    'M9.5 13l2.5 2.5 2.5-2.5',
  ],
};

export function ModelIcon({ name, size = 24, ...rest }) {
  return (
    <svg
      viewBox="0 0 24 24" width={size} height={size}
      fill="none" stroke="currentColor" strokeWidth={1.5}
      strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true" {...rest}
    >
      {PATHS[name].map((d) => <path key={d} d={d} />)}
    </svg>
  );
}
