# Historical GEAK tuning integration

The [original GEAK integration document](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/e2e_workflow/knowledge/tuning_skillset_integration.md)
describes that revision's phase ordering, role configuration, vendoring, validation and deployment
handoff. Its orchestrator arguments, sync scripts and evaluation directories belong to GEAK.

KernelWiki maintains the [tuning procedures](../../expert_recipes/tuning/README.md) as a local copy
reviewed through Git. The [upstream checksum manifest](tuning_skillset.manifest.sha256) is retained
as historical provenance. It does not validate or freeze the adapted local files; changes should
retain attribution and recheck the affected method on its intended environment.

Reusable conclusions are maintained in the [general corpus](../../corpus/decisions.md): verify
[deployment and graph engagement](../../corpus/decisions.md#method-live-binary-and-integration),
control [measurement drift](../../corpus/decisions.md#method-paired-measurement-controls), and
attribute [end-to-end changes](../../corpus/decisions.md#method-end-to-end-attribution) to the measured
combination. An accepted tuning change can alter the bottleneck, so re-profile before choosing the
next experiment.

New outcomes belong in operator or general corpus decisions. Keep deployable run artifacts, raw
reports and orchestration state in the private execution system. See the
[maintenance and contribution guide](../../../../tools/corpus/README.md).

Imported from AMD-AGI/GEAK at the revision linked above; this local summary replaces the duplicated
orchestration instructions and adds no new performance validation.
