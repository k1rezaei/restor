A comprehensive set of facts, referred to as $\mathcal{F}$ in the draft, is available at the following link: [facts.json](facts.json). Each entry in this dataset follows the format $(s, r, o)$, where $s$ represents the subject, $r$ denotes the relation, and $o$ signifies the object. For instance, here are two relations for the entity "Suthida":

```json
"Suthida": {
    "P27": {
        "question": "What is the country of citizenship of Suthida?",
        "answers": [
            "Thailand"
        ]
    },
    "P1412": {
        "question": "What languages does Suthida speak, write, or sign?",
        "answers": [
            "English",
            "Thai",
            "Southern Thai"
        ]
    },
}
```
Note that we may have several different answers for a pair of entity and relation. We follow the discussion in Appendix B of the draft and use facts in $\mathcal{F}$ to obtain corruption datasets, and follow discussions in Appendix H to obtain the SQuAD corruption dataset. These two can be found in the [Wikidata](wikidata/) and [SQuAD](SQuAD/) directories.


We evaluated model's knowledge on entities in the file: [entities.json](entities.json).
