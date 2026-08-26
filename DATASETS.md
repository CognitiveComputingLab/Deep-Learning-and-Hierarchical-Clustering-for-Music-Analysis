# Local dataset compatibility

## Directly usable: ABC/DCML string quartets

external/ABC contains 70 movements from 16 complete quartet works. Every
movement has a paired notes TSV and harmonies TSV, so all 70 can be used
directly for Greedy/DP clustering and DCML local-key Boundary F1.

The notes loader requires these columns:

- quarterbeats
- duration_qb
- midi

Boundary evaluation additionally requires a harmonies TSV containing:

- quarterbeats
- duration_qb
- localkey

Each of the 70 pieces also has an exact-name MuseScore file under
external/ABC/MS3. Pitch Scapes additionally needs MIDI. Only n11op95_01.mid is
currently exported in the project root, but MuseScore 4 is installed at:

    C:\Program Files\MuseScore 4\bin\MuseScore4.exe

Export another movement with resolved absolute paths (MuseScore's CLI is not
reliable with relative output paths):

    New-Item -ItemType Directory -Force results\midi | Out-Null
    $score = (Resolve-Path external\ABC\MS3\n01op18-1_01.mscx).Path
    $midiDir = (Resolve-Path results\midi).Path
    $target = Join-Path $midiDir n01op18-1_01.mid
    & 'C:\Program Files\MuseScore 4\bin\MuseScore4.exe' -o $target $score

Then pass that MIDI and the compatible notes TSV to a DP visualization script.
Exported MIDI timing must be checked against the TSV total duration, especially
for repeats and alternative endings.

The bundled export helper creates the default Op. 95 MIDI in the project root:

    powershell -ExecutionPolicy Bypass -File scripts\export_abc_midi.ps1

Export all 70 ABC scores to `results\midi` with:

    powershell -ExecutionPolicy Bypass -File scripts\export_abc_midi.ps1 -All

Pass `-MuseScore` if MuseScore 4 is installed at a non-default location, and
`-Piece piece_id` or `-Piece piece_a,piece_b` to export selected movements.

## Usable after conversion or pairing

### Pitchscapes repository examples

There are four MIDI files representing two duplicated example pieces: Bach's
BWV 846 Prelude and Beethoven's Op. 27 No. 2 sonata. They can provide a Pitch
Scape immediately, but the current DP loader cannot use MIDI as clustering
input. Convert MIDI notes to a TSV containing onset quarterbeats, duration in
quarterbeats, and MIDI pitch, or add a MIDI loader producing the same matrix
and bounds interface as load_pc_bins. They have no paired DCML local-key
annotations, so objective/tree visualization is possible but Boundary F1 is
not.

### Algomus data

The folder contains 60 Dezrann .dez annotation files and 20 MIDI files. The
MIDI files are AI-song melody examples and are not paired with the classical
.dez analyses. MIDI can be converted to notes TSV for clustering; .dez needs a
dedicated parser and a matching symbolic score before it can serve as reference
annotation. Do not treat unrelated MIDI and .dez files as a test pair.

### Taking Form

Taking Form contains 120 genuine hierarchical-form CSV annotations: 103
Beethoven sonata movements and 17 Mozart first movements. Its rows encode
measure, beat, and hierarchy columns from large to small units. This is a
promising future hierarchical ground truth, but it is not directly compatible
with the quarterbeat-based DP pipeline and the local checkout does not include
matching score files.

Integration requires a matching MusicXML/MuseScore/MIDI score, expansion of
repeats, conversion of measure+beat locations to absolute quarterbeats, and
construction of contiguous labelled interval trees from the hierarchy columns.
Evaluate these trees in a separate hierarchical experiment rather than mixing
them with DCML local-key Boundary F1.

## Conversion checks

For any new source, validate before evaluation:

1. Onsets and durations use quarterbeats and are finite and non-negative.
2. MIDI pitches are integer values and every sounding event has positive
   duration.
3. The score/MIDI and annotation timelines have the same repeat expansion.
4. Total duration and several known barlines align after conversion.
5. Train/validation/test splitting is performed by complete work, not movement.
6. Missing reference annotations are reported as visualization/objective-only,
   never as zero Boundary F1.
