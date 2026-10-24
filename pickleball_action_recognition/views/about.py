"""About: the research workflow, the classes and the current status."""

from components import ui

ui.page_header(
    "About",
    "Rally is a B.Tech research project on recognising pickleball strokes from video.",
)

# ---------------- The project ----------------
ui.html("""
<section class="rl-section" style="margin-top:0;">
<div><h2 class="rl-h2">The project</h2></div>
<div>
<p class="rl-copy">The aim is to tell five pickleball strokes apart by following how a player's body moves, rather than by looking at the ball. Pose keypoints are extracted from each frame, and a neural network learns what each stroke looks like over time.</p>
<p class="rl-copy">Uploaded clips are run through YOLOv8 pose estimation, and the resulting keypoints are classified by a CNN-BiLSTM. Accuracy depends on how much training data the model has seen, so treat predictions as indicative.</p>
</div>
</section>
""")



# ---------------- Target classes ----------------
ui.html(f"""
<section class="rl-section">
<div><h2 class="rl-h2">Target classes</h2></div>
{ui.strokes_list()}
</section>
""")