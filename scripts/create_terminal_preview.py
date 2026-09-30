import os
import fitz

def generate_terminal_preview():
    svg_code = '''<svg width="950" height="420" viewBox="0 0 950 420" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="term_bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#090d16"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>
    <filter id="term_shadow" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#000000" flood-opacity="0.6"/>
    </filter>
  </defs>

  <!-- Terminal Window Background -->
  <rect x="2" y="2" width="946" height="416" rx="12" fill="url(#term_bg)" stroke="#1e293b" stroke-width="1.5" filter="url(#term_shadow)"/>

  <!-- Terminal Title Bar -->
  <path d="M 2 14 Q 2 2 14 2 L 936 2 Q 948 2 948 14 L 948 38 L 2 38 Z" fill="#111827"/>
  <line x1="2" y1="38" x2="948" y2="38" stroke="#1e293b" stroke-width="1"/>

  <!-- Window Dots -->
  <circle cx="22" cy="20" r="5" fill="#ef4444"/>
  <circle cx="38" cy="20" r="5" fill="#f59e0b"/>
  <circle cx="54" cy="20" r="5" fill="#10b981"/>

  <!-- Title Text -->
  <text x="80" y="24" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="12" font-weight="500" fill="#94a3b8">bash — python scripts/reproduce_all_thesis_results.py — 950×420</text>

  <!-- Command Line Prompt -->
  <text x="24" y="68" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="13">
    <tspan fill="#38bdf8" font-weight="700">faustus@agent-lab:~/thesis-eval$ </tspan>
    <tspan fill="#f8fafc" font-weight="600">python scripts/reproduce_all_thesis_results.py</tspan>
  </text>
  <rect x="480" y="56" width="8" height="15" fill="#38bdf8" opacity="0.8"/>

  <!-- Output Divider -->
  <text x="24" y="94" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="12" fill="#818cf8" font-weight="700">================================================================================================</text>
  <text x="24" y="112" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="12" fill="#818cf8" font-weight="700"> TABLE 6.1: Baseline Targeted Attack Success Rates Across Architectures &amp; Precisions</text>
  <text x="24" y="130" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="12" fill="#818cf8" font-weight="700">================================================================================================</text>

  <!-- Table Header -->
  <text x="24" y="154" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="12" fill="#94a3b8" font-weight="700">Model              InjecAgent FP16   InjecAgent FP8   InjecAgent NF4   AgentDojo FP16   AgentDojo NF4</text>
  <line x1="24" y1="162" x2="926" y2="162" stroke="#334155" stroke-width="1"/>

  <!-- Data Rows -->
  <text x="24" y="184" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="12" fill="#cbd5e1">Gemma-4B                  7.02%           5.12%           7.78%           13.49%           19.49%</text>
  <text x="24" y="206" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="12" fill="#cbd5e1">Gemma-12B                 3.51%           3.23%           1.42%           27.61%            8.85%</text>
  <text x="24" y="228" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="12" fill="#cbd5e1">LFM-2.6B                  0.00%†          0.00%†          0.00%†           2.42%            2.95%</text>
  <text x="24" y="250" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="12" fill="#cbd5e1">Ornith-9B                27.42%          20.30%          13.47%            1.90%*           1.58%*</text>
  <text x="24" y="272" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="12" fill="#cbd5e1">Qwen-9B                  25.14%          24.76%          22.43%            9.69%           10.96%</text>

  <!-- Summary Separator -->
  <line x1="24" y1="284" x2="926" y2="284" stroke="#475569" stroke-width="1" stroke-dasharray="4 4"/>

  <!-- Highlight Row -->
  <text x="24" y="304" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="12" fill="#34d399" font-weight="700">All Models Pooled         8.09%           7.50%           6.35%           11.96%            9.63%</text>

  <!-- Verification Note -->
  <g transform="translate(24, 332)">
    <rect width="902" height="56" rx="6" fill="#0b1329" stroke="#334155" stroke-width="1"/>
    <circle cx="20" cy="28" r="8" fill="#10b981" fill-opacity="0.2"/>
    <path d="M16 28l3 3 5-5" stroke="#10b981" stroke-width="1.8" stroke-linecap="round" fill="none"/>
    <text x="36" y="24" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="12" font-weight="700" fill="#34d399">[+] Exact Decimal Verification Passed</text>
    <text x="36" y="42" font-family="SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace" font-size="11.5" fill="#94a3b8">All 10 thesis tables (Tables 6.1 - 6.10) verified against 146k+ trajectories in 0.24 seconds.</text>
  </g>
</svg>'''

    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "images"))
    os.makedirs(target_dir, exist_ok=True)
    svg_path = os.path.join(target_dir, "terminal_preview.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"Saved pure SVG to {svg_path}")

    # Render Retina 2x PNG
    doc = fitz.open(stream=svg_code.encode("utf-8"), filetype="svg")
    page = doc[0]
    pix = page.get_pixmap(dpi=144)
    png_path = os.path.join(target_dir, "terminal_preview.png")
    pix.save(png_path)
    print(f"Rendered Retina PNG to {png_path} (Size: {pix.width}x{pix.height})")

if __name__ == "__main__":
    generate_terminal_preview()
