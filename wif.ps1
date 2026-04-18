$PROJECT_ID="fifth-sunup-492317-g5"
$PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
$SA_EMAIL="github-action-deploy@fifth-sunup-492317-g5.iam.gserviceaccount.com"
$POOL_NAME="github-pool"
$PROVIDER_NAME="github-provider"

gcloud services enable iamcredentials.googleapis.com --project $PROJECT_ID
gcloud iam workload-identity-pools create $POOL_NAME --project=$PROJECT_ID --location="global" --display-name="GitHub Actions Pool"
gcloud iam workload-identity-pools providers create-oidc $PROVIDER_NAME --project=$PROJECT_ID --location="global" --workload-identity-pool=$POOL_NAME --display-name="GitHub Actions Provider" --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" --issuer-uri="https://token.actions.githubusercontent.com"

gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL --project=$PROJECT_ID --role="roles/iam.workloadIdentityUser" --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL_NAME/attribute.repository/NAT-SIRIRUANGBUN/SE_ML"

Write-Host "WIF_PROVIDER=projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL_NAME/providers/$PROVIDER_NAME"
Write-Host "WIF_SA=$SA_EMAIL"
