import { apiFetch } from "./client"

export interface OAuthProviderInfo {
  provider_id: string
  name: string
  flow: string
  description: string
  setup_command: string
}

export interface OAuthProvidersResponse {
  providers: OAuthProviderInfo[]
  note: string
}

export async function getOAuthProviders(): Promise<OAuthProvidersResponse> {
  return apiFetch("/v1/admin/oauth/providers")
}
