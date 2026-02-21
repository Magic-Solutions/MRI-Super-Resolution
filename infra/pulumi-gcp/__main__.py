import pulumi
import pulumi_gcp as gcp

cfg = pulumi.Config()

project = cfg.require("project")
region = cfg.get("region") or "us-central1"
project_info = gcp.organizations.get_project(project_id=project)
project_number = str(project_info.number)

training_data_bucket = cfg.require("trainingDataBucket")
wandb_secret_id = cfg.require("wandbSecretId")

runtime_sa_id = cfg.get("runtimeServiceAccountId") or "diamond-vertex-trainer"
repo_id = cfg.get("artifactRegistryRepoId") or "diamond-training"

# Hardware profile defaults consumed by submit scripts.
vertex_machine_type = cfg.get("vertexMachineType") or "a2-highgpu-1g"
vertex_accelerator_type = cfg.get("vertexAcceleratorType") or "NVIDIA_TESLA_A100"
vertex_accelerator_count = int(cfg.get("vertexAcceleratorCount") or "1")
vertex_boot_disk_size_gb = int(cfg.get("vertexBootDiskSizeGb") or "500")
artifact_bucket_name = f"{project}-diamond-vertex-artifacts"

runtime_sa = gcp.serviceaccount.Account(
    "runtime-sa",
    project=project,
    account_id=runtime_sa_id,
    display_name="DIAMOND Vertex runtime service account",
)

artifact_repo = gcp.artifactregistry.Repository(
    "training-repo",
    project=project,
    location=region,
    repository_id=repo_id,
    format="DOCKER",
    description="Container images for DIAMOND Vertex training jobs",
)

artifact_bucket_resource = gcp.storage.Bucket(
    "artifact-bucket",
    project=project,
    name=artifact_bucket_name,
    location=region,
    storage_class="STANDARD",
    force_destroy=False,
    uniform_bucket_level_access=True,
)

gcp.storage.BucketIAMMember(
    "training-data-viewer",
    bucket=training_data_bucket,
    role="roles/storage.objectViewer",
    member=runtime_sa.email.apply(lambda email: f"serviceAccount:{email}"),
)

gcp.storage.BucketIAMMember(
    "artifact-writer",
    bucket=artifact_bucket_resource.name,
    role="roles/storage.objectAdmin",
    member=runtime_sa.email.apply(lambda email: f"serviceAccount:{email}"),
    opts=pulumi.ResourceOptions(depends_on=[artifact_bucket_resource]),
)

gcp.secretmanager.SecretIamMember(
    "wandb-secret-accessor",
    project=project,
    secret_id=wandb_secret_id,
    role="roles/secretmanager.secretAccessor",
    member=runtime_sa.email.apply(lambda email: f"serviceAccount:{email}"),
)

pulumi.export("project", project)
pulumi.export("region", region)
pulumi.export("runtimeServiceAccountEmail", runtime_sa.email)
pulumi.export("artifactBucket", artifact_bucket_resource.name)
pulumi.export("artifactRepository", artifact_repo.repository_id)
pulumi.export(
    "artifactImageBase",
    pulumi.Output.all(project, region, artifact_repo.repository_id).apply(
        lambda args: f"{args[1]}-docker.pkg.dev/{args[0]}/{args[2]}"
    ),
)
pulumi.export("vertexMachineType", vertex_machine_type)
pulumi.export("vertexAcceleratorType", vertex_accelerator_type)
pulumi.export("vertexAcceleratorCount", vertex_accelerator_count)
pulumi.export("vertexBootDiskSizeGb", vertex_boot_disk_size_gb)

# Cloud Build service account permissions for image push/logging.
# Depending on project/org, builds can run as one of these principals.
cloud_build_service_accounts = [
    f"{project_number}-compute@developer.gserviceaccount.com",
    f"{project_number}@cloudbuild.gserviceaccount.com",
]

for i, sa_email in enumerate(cloud_build_service_accounts):
    gcp.artifactregistry.RepositoryIamMember(
        f"cloudbuild-artifact-writer-{i}",
        project=project,
        location=region,
        repository=artifact_repo.repository_id,
        role="roles/artifactregistry.writer",
        member=f"serviceAccount:{sa_email}",
    )
    gcp.projects.IAMMember(
        f"cloudbuild-log-writer-{i}",
        project=project,
        role="roles/logging.logWriter",
        member=f"serviceAccount:{sa_email}",
    )
    gcp.projects.IAMMember(
        f"cloudbuild-storage-viewer-{i}",
        project=project,
        role="roles/storage.objectViewer",
        member=f"serviceAccount:{sa_email}",
    )

pulumi.export("cloudBuildServiceAccounts", cloud_build_service_accounts)
