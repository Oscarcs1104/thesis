"""Model package. Import concrete classes from their modules, e.g.
``from model.model import MultimodalModel`` or
``from model.conditional_generator import ConditionalSmilesGenerator`` -- this
package does not eagerly import them, so the standalone generator does not drag
in the predictor's graph/text encoder stack (or RDKit)."""
