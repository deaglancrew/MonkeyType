from typing import List


def _group_generated_classes(module: "ModuleStub"):
    duplicates = {}
    for generated_class_stub in module.generated_class_stubs:
        template = generated_class_stub.get_template()
        if template not in duplicates:
            duplicates[template] = []
        duplicates[template].append(generated_class_stub)
    return duplicates


def _reduce_duplicates(stubs: List["ClassStub"]):
    sorted_stubs = sorted(stubs, key=lambda stub: stub.name)
    current_stub = None
    for stub in sorted_stubs:
        if current_stub is None:
            current_stub = stub
        else:
            for ref in stub.references:
                current_stub.add_ref(ref)
    current_stub.update_refs()
    return current_stub


def dedupe_module(module: "ModuleStub"):
    new_classes = []
    for class_group, duplicates in _group_generated_classes(module).items():
        condensed = _reduce_duplicates(duplicates)
        new_classes.append(condensed)
    module.generated_class_stubs = new_classes
    return module
