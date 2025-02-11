az login

View subscriptions:
az account subscription list

Create a resource group:
az group create --name jobstore-ai --location westus2

All locations for resource groups:
az account list-locations


Install CLI extension for azure Kubernetes service:
az extension add --name aks-preview
az extension update --name aks-preview

Enable GPU support for AKS (15 mins - installs GPUDedicatedVHDPreview from Microsoft.ContainerService):
az feature register --namespace "Microsoft.ContainerService" --name "GPUDedicatedVHDPreview"

View status of the installation:
az feature show --namespace "Microsoft.ContainerService" --name "GPUDedicatedVHDPreview"

Aside - all features in that namespace:
az feature list --namespace Microsoft.ContainerService -o table

Or see them via:
- Going to Subscriptions
- Selecting your subscription
- Clicking on "Preview features" under Settings
- Filtering the list by namespace


Register subscription to work with Azure Container Service resources (10 mins - one-time process per subscription):
az provider register --namespace Microsoft.ContainerService

Monitor provider registration:
az provider show --n Microsoft.ContainerService --query "registrationState"

Also need to register the Microsoft.Insights provider (resource provider for Azure Monitor):
az provider register --namespace Microsoft.Insights
az provider show -n Microsoft.Insights --query="registrationState"

Also need Microsoft.Compute for managing VM quotas:
az provider register --namespace Microsoft.Compute
az provider show -n Microsoft.Compute --query="registrationState"

Allows disable SSH and avoiding ssh enabled warning:
az feature register --namespace "Microsoft.ContainerService" --name "DisableSSHPreview"
az provider show -n Microsoft.ContainerService --query="registrationState"
az provider register -n Microsoft.ContainerService
^ this reloads provider registration to recognize the newly registered feature
az provider show --namespace Microsoft.ContainerService --query registrationState
^ check registration again

Create base AKS cluster (few mins):
az aks create `
    --resource-group jobstore-ai `
    --name AKSCluster `
    --node-count 1 `
    --enable-addons monitoring `
    --ssh-access disabled `
    --node-vm-size Standard_D2s_v3 `
    --node-osdisk-type Ephemeral `
    --node-osdisk-size 32 `
    --max-pods 30

--node-count 1: Minimum required nodes for system node pool. Cannot be 0 for initial creation. User node pools can scale to 0. Critical for base cluster functionality.
--node-vm-size Standard_D2s_v3: Most economical VM size supporting AKS (default)
--node-osdisk-type Ephemeral: Uses VM's local storage, included in VM price, no additional storage costs. Data lost on node restart/reallocation. Alternative: Managed (persistent but costs more). Perfect for AKS as nodes shouldn't store persistent data.
--node-osdisk-size 32: Minimum size in GB for OS disk. With Ephemeral, limited by VM's cache/temporary storage size. Cannot exceed VM's local storage limits. Standard_B2s supports this minimum size.
--max-pods 30: Limits pods per node, optimized for system workloads. Default is 250 which is excessive for most uses. Lower number improves node stability and resource management.
--ssh-access disabled: Disables SSH access to nodes for security. Alternative: enabled (needed only for direct node debugging). Best practice to keep disabled unless specifically needed.
--enable-addons monitoring: Enables Azure Monitor for containers. Alternative addons: http_application_routing, virtual-node. Essential for cluster monitoring and diagnostics.

NOTE: you cannot configure storage type for system node - it depends on node-vm-size.
NOTE: you also cannot get it to use spot pricing due to stability requirements

This will start it with a single system node on Standard_DS2_v2 VM (this is not for workloads), which will immediately start charging $$.

NOTE:
For ephemeral storage (won't charge for storage this way):
--node-osdisk-type Ephemeral

Otherwise set to (typically not needed):
--node-osdisk-type Managed

== To stop the whole cluster:
az aks stop --name AKSCluster --resource-group jobstore-ai

To start the cluster back up (wait 15 min between restarts?):
az aks start --name AKSCluster --resource-group jobstore-ai

Delete Cluster:
az aks delete --name AKSCluster --resource-group jobstore-ai --yes

==Subsciribe to pay as you go  $$==

To monitor $$:
https://dotnet.microsoft.com/en-us/download/dotnet/thank-you/sdk-9.0.102-windows-x64-installer
dotnet tool install --global azure-cost-cli
# View accumulated costs
azure-cost accumulatedCost -s SUBSCRIPTION_ID
# View costs by resource
azure-cost costByResource -o text -s SUBSCRIPTION_ID
# View all resources
az resource list -o table
az resource list --resource-group jobstore-ai -o table
# Cost analysis
https://portal.azure.com/#view/Microsoft_Azure_CostManagement/Menu/~/costanalysis/open/costanalysisv3/openedBy/AzurePortal
# All resources
https://portal.azure.com/#browse/all
# VMs
az vm list -d --query "[].{Name:name, PowerState:powerState}" -o table
https://portal.azure.com/#browse/Microsoft.Compute%2FVirtualMachines
# Active storage accounts:
az storage account list --query "[].{Name:name, ResourceGroup:resourceGroup}" -o table
# View all AKS clusters and their statuses:
az aks list --query "[].{Name:name, ResourceGroup:resourceGroup, PowerState:powerState.code}" -o table
# To see app services:
az webapp list --query "[].{Name:name, State:state}" -o table

Check cluster status:
az aks list --resource-group jobstore-ai -o table

View aks cluster details:
az aks show --resource-group jobstore-ai --name AKSCluster

View node pools on AKS cluster:
az aks nodepool list --resource-group jobstore-ai --cluster-name AKSCluster

View node pool details:
az aks nodepool show --resource-group jobstore-ai --cluster-name AKSCluster --name nodepool1


Add GPU node pool (it will not start any nodes - initialized to 0)
az aks nodepool add `
    --resource-group jobstore-ai `
    --cluster-name AKSCluster `
    --name gpunodepool `
    --node-count 0 `
    --node-vm-size Standard_NC4as_T4_v3 `
    --priority Spot `
    --eviction-policy Delete `
    --spot-max-price -1 `
    --node-taints sku=gpu:NoSchedule `
    --skip-gpu-driver-install `
    --ssh-access disabled `
    --os-sku AzureLinux `
    --node-osdisk-type Ephemeral `
    --node-osdisk-size 64

--resource-group: The Azure resource group containing your AKS cluster
--cluster-name: Name of your existing AKS cluster where this pool will be added
--name: Unique name for this GPU-enabled node pool
--node-count: Initial number of nodes (0 to minimize costs)
--node-vm-size: Standard_NC4as_T4_v3 - cheapest GPU VM with 4 CPU, 28GB RAM, 8GB NVIDIA T4 GPU
--priority: Uses spot instances for significant cost savings (60-90% cheaper)
--eviction-policy: Immediately deletes VM on spot eviction instead of deallocating
--spot-max-price: -1 means use current spot price without maximum cap
--enable-cluster-autoscaler: Automatically scales nodes based on workload demand
--min-count: Minimum node count (can scale to 0 when idle to save costs)
--max-count: Maximum node count (limits maximum cost exposure)
--node-taints: Ensures only GPU workloads run on these expensive nodes
--skip-gpu-driver-install: Skips default driver installation to use NVIDIA GPU Operator
--ssh-access: Disables SSH access for security
--os-sku: Uses lightweight AzureLinux optimized for containers
--node-osdisk-type: Uses ephemeral storage (VM's local storage) instead of managed disk
--node-osdisk-size: Sets OS disk to 64GB (must be ≤75% of VM's temporary storage of 176GB)


Install Nvidia GPU Operator to manage drivers (since we skipped driver install in previous step - preferred method)

helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
    && helm repo update

Get aks credentials for the cluster:
az aks get-credentials --resource-group jobstore-ai --name AKSCluster

On WSL, may need to:
- mkdir -p ~/.kube
- cp /mnt/c/Users/<USER>/.kube/config ~/.kube/config
- Edit ~/.profile: add `export KUBECONFIG=~/.kube/config`
- source ~/.profile

Check kubectl is working:
kubectl get nodes

Install the operator:
helm install --wait --generate-name \
    -n gpu-operator --create-namespace \
    nvidia/gpu-operator \
    --version=v24.9.2

Can see it in Kubernetes resources -> Workloads, or run

helm list -n gpu-operator

Start a GPU-enabled node:

First, disabled automatic scaling (better to control it manually ourselves):
az aks nodepool update \
    --resource-group jobstore-ai \
    --cluster-name AKSCluster \
    --name gpunodepool \
    --disable-cluster-autoscaler

Scale to 1 node count for GPU pool:
az aks nodepool scale \
    --resource-group jobstore-ai \
    --cluster-name AKSCluster \
    --name gpunodepool \
    --node-count 1

If you get:
`(ErrCode_InsufficientVCPUQuota) Insufficient vcpu quota requested 4, remaining 0 for family Standard NCASv3_T4 Family for region westus.`

Then request quota increase at:
https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/overview

To check quota status:
az quota show --resource-name "Standard NCASv3_T4 Family" --scope "subscriptions/xxx/providers/Microsoft.Compute/locations/westus2"
az quota show --resource-name "Standard NCASv3_T4 Family" --scope "subscriptions/xxx/providers/Microsoft.Compute/locations/westus"

See active nodes (should be 2 - system node and gpu node):

kubectl get nodes

To check if we're being provisioned through spot pricing:

az aks nodepool list --resource-group jobstore-ai --cluster-name AKSCluster --query "[].{Name:name, VMSize:vmSize, ScaleSetPriority:scaleSetPriority, SpotMaxPrice:spotMaxPrice}" -o table