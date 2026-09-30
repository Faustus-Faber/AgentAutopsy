import os
import fitz

def generate_pixel_perfect_banner():
    W, H = 1200, 350
    
    svg_code = f'''<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- Background Gradient -->
    <linearGradient id="bg_grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#070A12"/>
      <stop offset="50%" stop-color="#0B1120"/>
      <stop offset="100%" stop-color="#0F172A"/>
    </linearGradient>

    <!-- Ambient Glows -->
    <radialGradient id="glow_cyan" cx="85%" cy="30%" r="50%">
      <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#38BDF8" stop-opacity="0.0"/>
    </radialGradient>
    <radialGradient id="glow_indigo" cx="15%" cy="20%" r="50%">
      <stop offset="0%" stop-color="#6366F1" stop-opacity="0.10"/>
      <stop offset="100%" stop-color="#6366F1" stop-opacity="0.0"/>
    </radialGradient>

    <!-- Top Accent Gradient -->
    <linearGradient id="accent_bar" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="50%" stop-color="#6366F1"/>
      <stop offset="100%" stop-color="#A855F7"/>
    </linearGradient>

    <!-- Grid Pattern -->
    <pattern id="dot_grid" width="28" height="28" patternUnits="userSpaceOnUse">
      <circle cx="2" cy="2" r="0.8" fill="#334155" fill-opacity="0.4"/>
    </pattern>
  </defs>

  <!-- 1. FULL-BLEED BASE: 100% covers canvas, zero white corners -->
  <rect width="{W}" height="{H}" fill="#070A12"/>

  <!-- 2. CANVAS MAIN RECTANGLE WITH CRISP SUBTLE BORDER -->
  <rect x="0" y="0" width="{W}" height="{H}" fill="url(#bg_grad)"/>
  <rect x="0" y="0" width="{W}" height="{H}" fill="url(#dot_grid)"/>
  <rect x="0" y="0" width="{W}" height="{H}" fill="url(#glow_cyan)"/>
  <rect x="0" y="0" width="{W}" height="{H}" fill="url(#glow_indigo)"/>
  <rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" fill="none" stroke="#1E293B" stroke-width="1"/>

  <!-- Top Decorative Accent Line -->
  <rect x="0" y="0" width="{W}" height="2.5" fill="url(#accent_bar)"/>

  <!-- ======================================================== -->
  <!-- LEFT COLUMN: THESIS BRANDING & METRIC CARDS              -->
  <!-- ======================================================== -->

  <!-- Institutional Pre-Header Badge -->
  <g transform="translate(46, 26)">
    <rect width="320" height="26" rx="13" fill="#1E1B4B" fill-opacity="0.75" stroke="#4F46E5" stroke-width="1" stroke-opacity="0.6"/>
    <circle cx="14" cy="13" r="3.5" fill="#38BDF8"/>
    <text x="26" y="17" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-size="11" font-weight="700" letter-spacing="0.06em" fill="#C7D2FE">BRAC UNIVERSITY • CSE THESIS 2026</text>
  </g>

  <!-- Master Title: AgentAutopsy (Solid #FFFFFF, 100% Reliable Render) -->
  <text x="46" y="98" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-size="44" font-weight="900" letter-spacing="-0.03em" fill="#FFFFFF">AgentAutopsy</text>

  <!-- Core Thesis Tagline -->
  <text x="46" y="134" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-size="19" font-weight="700" letter-spacing="-0.01em" fill="#38BDF8">Low Attack Success Is Not Security</text>

  <!-- Official Thesis Subtitle -->
  <text x="46" y="162" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-size="14" font-weight="400" fill="#E2E8F0">Diagnosing Indirect Prompt-Injection Failures in Small Tool-Using Agents</text>

  <!-- Empirical Scope Summary -->
  <text x="46" y="186" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-size="12" font-weight="400" fill="#94A3B8">Forensic evaluation across 146,000+ trajectories exposing syntax collapse, inattention &amp; hollow defenses.</text>

  <!-- 4 ISOLATED METRIC CARDS (Separate Icon Column + Separate Text Column = ZERO OVERLAP) -->
  <g transform="translate(46, 218)">
    
    <!-- CARD 1: 11 Models (2B to 14B Cohort) -->
    <g transform="translate(0, 0)">
      <rect width="154" height="56" rx="8" fill="#0C1425" stroke="#1E293B" stroke-width="1"/>
      <!-- Dedicated Icon Well (x: 10..42) -->
      <rect x="10" y="12" width="32" height="32" rx="6" fill="#131F38" stroke="#1E293B" stroke-width="0.8"/>
      <!-- Microchip Vector Path centered at (26, 28) -->
      <path d="M22 24h8v8h-8z M26 20v4 M26 32v4 M22 28h-4 M30 28h4" stroke="#38BDF8" stroke-width="1.6" stroke-linecap="round" fill="none"/>
      <!-- Text starts strictly at x=50 (ZERO OVERLAP) -->
      <text x="50" y="27" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11.5" font-weight="800" fill="#FFFFFF">11 MODELS</text>
      <text x="50" y="42" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10.5" font-weight="600" fill="#38BDF8">2B to 14B Cohort</text>
    </g>

    <!-- CARD 2: Dual Benchmark (InjecAgent + AgentDojo) -->
    <g transform="translate(162, 0)">
      <rect width="154" height="56" rx="8" fill="#0C1425" stroke="#1E293B" stroke-width="1"/>
      <!-- Dedicated Icon Well (x: 10..42) -->
      <rect x="10" y="12" width="32" height="32" rx="6" fill="#131F38" stroke="#1E293B" stroke-width="0.8"/>
      <!-- Shield Vector Path centered at (26, 28) -->
      <path d="M21 24l5-2.5 5 2.5v4c0 3.5-2.5 6-5 7-2.5-1-5-3.5-5-7v-4z" stroke="#34D399" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
      <!-- Text starts strictly at x=50 (ZERO OVERLAP) -->
      <text x="50" y="27" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11.5" font-weight="800" fill="#FFFFFF">DUAL SUITE</text>
      <text x="50" y="42" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10.5" font-weight="600" fill="#34D399">InjecAgent + Dojo</text>
    </g>

    <!-- CARD 3: 33,000 Audited Cases (ATOM Behavioral Matrix) -->
    <g transform="translate(324, 0)">
      <rect width="154" height="56" rx="8" fill="#0C1425" stroke="#1E293B" stroke-width="1"/>
      <!-- Dedicated Icon Well (x: 10..42) -->
      <rect x="10" y="12" width="32" height="32" rx="6" fill="#131F38" stroke="#1E293B" stroke-width="0.8"/>
      <!-- Microscope Vector Path centered at (26, 28) -->
      <path d="M21 23l4 4 M26 21l-3-3 M24 24l3 3 M20 32h12" stroke="#F59E0B" stroke-width="1.6" stroke-linecap="round" fill="none"/>
      <!-- Text starts strictly at x=50 (ZERO OVERLAP) -->
      <text x="50" y="27" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11.5" font-weight="800" fill="#FFFFFF">33k AUDITED</text>
      <text x="50" y="42" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10.5" font-weight="600" fill="#FBBF24">ATOM Taxonomy</text>
    </g>

    <!-- CARD 4: 146,000+ Raw Trajectories (FP16 / FP8 / NF4) -->
    <g transform="translate(486, 0)">
      <rect width="154" height="56" rx="8" fill="#0C1425" stroke="#1E293B" stroke-width="1"/>
      <!-- Dedicated Icon Well (x: 10..42) -->
      <rect x="10" y="12" width="32" height="32" rx="6" fill="#131F38" stroke="#1E293B" stroke-width="0.8"/>
      <!-- Database Vector Path centered at (26, 28) -->
      <path d="M20 23c0-1.2 2.7-2 6-2s6 0.8 6 2v6c0 1.2-2.7 2-6 2s-6-0.8-6-2v-6z M20 26c0 1.2 2.7 2 6 2s6-0.8 6-2" stroke="#818CF8" stroke-width="1.6" stroke-linecap="round" fill="none"/>
      <!-- Text starts strictly at x=50 (ZERO OVERLAP) -->
      <text x="50" y="27" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11.5" font-weight="800" fill="#FFFFFF">146k+ RUNS</text>
      <text x="50" y="42" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10.5" font-weight="600" fill="#A5B4FC">FP16 • FP8 • NF4</text>
    </g>
  </g>

  <!-- Secondary Metadata Line Below Cards -->
  <g transform="translate(46, 292)">
    <circle cx="6" cy="6" r="3" fill="#10B981"/>
    <text x="16" y="10" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="500" fill="#64748B">Exact Decimal Replication (<tspan fill="#10B981" font-weight="600">0.24s</tspan>) • 8 Controlled Causal Interventions • Public Kaggle Parquet Release</text>
  </g>

  <!-- ======================================================== -->
  <!-- RIGHT COLUMN: AUTHENTIC THESIS FINDINGS CONSOLE          -->
  <!-- ======================================================== -->

  <g transform="translate(692, 26)">
    <!-- Console Outer Frame (w=462, h=296) -->
    <rect width="462" height="296" rx="10" fill="#0A0F1D" stroke="#1E293B" stroke-width="1.2"/>

    <!-- Window Header -->
    <path d="M 0 10 Q 0 0 10 0 L 452 0 Q 462 0 462 10 L 462 32 L 0 32 Z" fill="#0F172A"/>
    <line x1="0" y1="32" x2="462" y2="32" stroke="#1E293B" stroke-width="1"/>

    <!-- Window Dots -->
    <circle cx="18" cy="16" r="4" fill="#EF4444"/>
    <circle cx="30" cy="16" r="4" fill="#F59E0B"/>
    <circle cx="42" cy="16" r="4" fill="#10B981"/>
    <text x="60" y="20" font-family="SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="10.5" font-weight="600" fill="#94A3B8">KEY EMPIRICAL FINDINGS // THESIS CHAPTER 6</text>

    <!-- Finding 1: Silent Inattention (Thesis §6.3.1) -->
    <g transform="translate(16, 44)">
      <rect width="430" height="52" rx="6" fill="#121A2D" stroke="#F59E0B" stroke-width="0.8" stroke-opacity="0.4"/>
      <circle cx="14" cy="16" r="4" fill="#F59E0B"/>
      <text x="24" y="20" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11.5" font-weight="700" fill="#FDE68A">SILENT INATTENTION DOMINATES (&gt;65%)</text>
      <text x="14" y="38" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10.5" font-weight="400" fill="#CBD5E1">Over 65% of nominal defenses are passive unrecognized task continuation.</text>
    </g>

    <!-- Finding 2: Perception-Compliance Dissociation (Thesis §6.3.4) -->
    <g transform="translate(16, 104)">
      <rect width="430" height="52" rx="6" fill="#1A1426" stroke="#EF4444" stroke-width="0.8" stroke-opacity="0.4"/>
      <circle cx="14" cy="16" r="4" fill="#EF4444"/>
      <text x="24" y="20" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11.5" font-weight="700" fill="#FCA5A5">PERCEPTION–COMPLIANCE DISSOCIATION (PCD)</text>
      <text x="14" y="38" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10.5" font-weight="400" fill="#CBD5E1">Models detect injections in CoT (28–47%), yet comply anyway in 41–74%.</text>
    </g>

    <!-- Finding 3: Untriggered Default & Authority Shift (Thesis §6.5.2) -->
    <g transform="translate(16, 164)">
      <rect width="430" height="52" rx="6" fill="#0C1B2E" stroke="#38BDF8" stroke-width="0.8" stroke-opacity="0.4"/>
      <circle cx="14" cy="16" r="4" fill="#38BDF8"/>
      <text x="24" y="20" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11.5" font-weight="700" fill="#BAE6FD">THE UNTRIGGERED DEFAULT (+34.98pp SURGE)</text>
      <text x="14" y="38" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10.5" font-weight="400" fill="#CBD5E1">System authorization shifts baseline ASR from 8.09% up to 43.07% (up to 70.9%).</text>
    </g>

    <!-- Verdict Summary Footer (Thesis §6.6) -->
    <g transform="translate(16, 224)">
      <rect width="430" height="56" rx="6" fill="#0B132B" stroke="#6366F1" stroke-width="1" stroke-opacity="0.5"/>
      <path d="M14 28c0-5 4-9 9-9s9 4 9 9-4 9-9 9-9-4-9-9z M23 23v6 M23 33h.01" stroke="#818CF8" stroke-width="1.6" stroke-linecap="round" fill="none"/>
      <text x="38" y="24" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="800" fill="#C7D2FE">THESIS VERDICT: INCOMPETENCE MASQUERADING AS ROBUSTNESS</text>
      <text x="38" y="42" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10" font-weight="400" fill="#94A3B8">Low ASR is driven by syntax collapse and passive inattention, not security alignment.</text>
    </g>
  </g>
</svg>'''

    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "images"))
    os.makedirs(target_dir, exist_ok=True)

    svg_path = os.path.join(target_dir, "banner.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"Saved pristine SVG to {svg_path}")

    # Render Retina 2x PNG (2400 x 700)
    doc = fitz.open(stream=svg_code.encode("utf-8"), filetype="svg")
    page = doc[0]
    pix = page.get_pixmap(dpi=144, alpha=False)
    png_path = os.path.join(target_dir, "banner.png")
    pix.save(png_path)
    print(f"Rendered Retina PNG to {png_path} (Size: {pix.width}x{pix.height})")

if __name__ == "__main__":
    generate_pixel_perfect_banner()
