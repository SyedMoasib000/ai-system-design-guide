# Access Control for LLM Systems

Secure access control is essential for multi-user and multi-tenant LLM applications. This chapter covers authentication, authorization, and data isolation patterns.

## Table of Contents

- [Access Control Requirements](#access-control-requirements)
- [Authentication Patterns](#authentication-patterns)
- [Authorization Models](#authorization-models)
- [Tenant Isolation](#tenant-isolation)
- [API Key Management](#api-key-management)
- [Audit and Compliance](#audit-and-compliance)
- [Interview Questions](#interview-questions)
- [References](#references)

---

## Access Control Requirements

### Security Dimensions

| Dimension | Description | Controls |
|-----------|-------------|----------|
| **Authentication** | Who is making the request? | API keys, OAuth, JWT |
| **Authorization** | What can they do? | RBAC, ABAC, policies |
| **Isolation** | What data can they see? | Tenant filtering, encryption |
| **Audit** | What did they do? | Logging, compliance reports |

### LLM-Specific Concerns

| Concern | Risk | Mitigation |
|---------|------|------------|
| Prompt injection | Bypass access controls | Input validation |
| Data leakage | Cross-tenant exposure | Strict filtering |
| Model output | Expose protected info | Output filtering |
| Context pollution | Inject unauthorized data | Context validation |

---

## Authentication Patterns

### API Key Authentication

```python
class APIKeyAuthenticator:
    def __init__(self, key_store):
        self.key_store = key_store
    
    async def authenticate(self, api_key: str) -> AuthResult:
        if not api_key:
            return AuthResult(authenticated=False, error="Missing API key")
        
        # Hash the key for lookup
        key_hash = self.hash_key(api_key)
        
        # Look up in store
        key_record = await self.key_store.get(key_hash)
        
        if not key_record:
            return AuthResult(authenticated=False, error="Invalid API key")
        
        if key_record.expired:
            return AuthResult(authenticated=False, error="Expired API key")
        
        if key_record.revoked:
            return AuthResult(authenticated=False, error="Revoked API key")
        
        return AuthResult(
            authenticated=True,
            user_id=key_record.user_id,
            tenant_id=key_record.tenant_id,
            scopes=key_record.scopes
        )
    
    def hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()
```

### JWT with Scopes

```python
class JWTAuthenticator:
    def __init__(self, public_key: str):
        self.public_key = public_key
    
    async def authenticate(self, token: str) -> AuthResult:
        try:
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=["RS256"],
                audience="llm-api"
            )
            
            return AuthResult(
                authenticated=True,
                user_id=payload["sub"],
                tenant_id=payload.get("tenant_id"),
                scopes=payload.get("scopes", []),
                expires_at=datetime.fromtimestamp(payload["exp"])
            )
        except jwt.ExpiredSignatureError:
            return AuthResult(authenticated=False, error="Token expired")
        except jwt.InvalidTokenError as e:
            return AuthResult(authenticated=False, error=str(e))
```

---

## Authorization Models

### Role-Based Access Control (RBAC)

```python
class RBACAuthorizer:
    ROLE_PERMISSIONS = {
        "admin": ["*"],
        "developer": ["generate", "embed", "fine_tune", "read_metrics"],
        "user": ["generate", "embed"],
        "viewer": ["read_metrics"]
    }
    
    def authorize(self, user: User, action: str) -> bool:
        permissions = self.ROLE_PERMISSIONS.get(user.role, [])
        
        if "*" in permissions:
            return True
        
        return action in permissions
```

### Attribute-Based Access Control (ABAC)

```python
class ABACAuthorizer:
    def __init__(self, policy_engine):
        self.policy_engine = policy_engine
    
    async def authorize(
        self,
        subject: dict,       # Who (user attributes)
        action: str,         # What (operation)
        resource: dict,      # On what (resource attributes)
        context: dict        # When/where (environmental)
    ) -> AuthzResult:
        # Evaluate all applicable policies
        policies = await self.policy_engine.get_policies(action)
        
        for policy in policies:
            result = policy.evaluate(subject, action, resource, context)
            if result == PolicyResult.DENY:
                return AuthzResult(allowed=False, reason=policy.name)
            if result == PolicyResult.ALLOW:
                return AuthzResult(allowed=True)
        
        return AuthzResult(allowed=False, reason="No matching policy")
```

### Model-Level Permissions

```python
class ModelAccessControl:
    MODEL_TIERS = {
        "gpt-4o": ["enterprise", "professional"],
        "gpt-4o-mini": ["enterprise", "professional", "starter"],
        "claude-3.5-sonnet": ["enterprise"],
        "claude-3.5-haiku": ["enterprise", "professional", "starter"]
    }
    
    def can_access_model(self, user: User, model: str) -> bool:
        allowed_tiers = self.MODEL_TIERS.get(model, [])
        return user.tier in allowed_tiers
    
    def get_available_models(self, user: User) -> list[str]:
        return [
            model for model, tiers in self.MODEL_TIERS.items()
            if user.tier in tiers
        ]
```

---

## Tenant Isolation

### Data Isolation Patterns

```python
class TenantIsolatedVectorStore:
    def __init__(self, vector_db):
        self.db = vector_db
    
    async def search(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int = 10
    ) -> list[dict]:
        # CRITICAL: Always filter by tenant_id at database level
        results = await self.db.search(
            query_vector=query_embedding,
            top_k=top_k,
            filter={"tenant_id": {"$eq": tenant_id}}  # Mandatory filter
        )
        
        return results
    
    async def insert(
        self,
        tenant_id: str,
        documents: list[dict]
    ):
        # CRITICAL: Always include tenant_id in metadata
        for doc in documents:
            doc["metadata"]["tenant_id"] = tenant_id
        
        await self.db.insert(documents)
```

### Prompt Isolation

```python
class TenantAwarePromptBuilder:
    def build_prompt(
        self,
        tenant_id: str,
        user_query: str,
        context: list[dict]
    ) -> str:
        # Verify all context belongs to tenant
        for doc in context:
            if doc.get("tenant_id") != tenant_id:
                raise SecurityError("Cross-tenant context detected")
        
        # Build isolated prompt
        return f"""
[Tenant: {tenant_id}]
Context from tenant documents:
{self.format_context(context)}

User query: {user_query}
"""
```

### Cache Isolation

```python
class TenantIsolatedCache:
    def __init__(self, cache_backend):
        self.cache = cache_backend
    
    def _scoped_key(self, tenant_id: str, key: str) -> str:
        return f"tenant:{tenant_id}:{key}"
    
    async def get(self, tenant_id: str, key: str) -> any:
        return await self.cache.get(self._scoped_key(tenant_id, key))
    
    async def set(self, tenant_id: str, key: str, value: any, ttl: int = 3600):
        await self.cache.set(
            self._scoped_key(tenant_id, key),
            value,
            ttl=ttl
        )
```

---

## API Key Management

### Key Lifecycle

```python
class APIKeyManager:
    KEY_PREFIX = "llm_"
    
    async def create_key(
        self,
        user_id: str,
        tenant_id: str,
        name: str,
        scopes: list[str],
        expires_in_days: int = 365
    ) -> APIKey:
        # Generate secure key
        raw_key = self.KEY_PREFIX + secrets.token_urlsafe(32)
        key_hash = self.hash_key(raw_key)
        
        # Store metadata (not the raw key)
        key_record = APIKeyRecord(
            id=generate_id(),
            hash=key_hash,
            user_id=user_id,
            tenant_id=tenant_id,
            name=name,
            scopes=scopes,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=expires_in_days)
        )
        
        await self.store.save(key_record)
        
        # Return raw key only once (not stored)
        return APIKey(
            id=key_record.id,
            key=raw_key,  # Only returned on creation
            name=name,
            scopes=scopes,
            expires_at=key_record.expires_at
        )
    
    async def revoke_key(self, key_id: str, reason: str):
        await self.store.update(key_id, {
            "revoked": True,
            "revoked_at": datetime.now(),
            "revoke_reason": reason
        })
        
        await self.audit_log.log("api_key_revoked", {
            "key_id": key_id,
            "reason": reason
        })
```

### Key Rotation

```python
class KeyRotator:
    async def rotate_key(self, old_key_id: str) -> APIKey:
        old_key = await self.key_store.get(old_key_id)
        
        # Create new key with same permissions
        new_key = await self.key_manager.create_key(
            user_id=old_key.user_id,
            tenant_id=old_key.tenant_id,
            name=f"{old_key.name} (rotated)",
            scopes=old_key.scopes
        )
        
        # Grace period: old key still works temporarily
        await self.key_store.update(old_key_id, {
            "deprecated": True,
            "deprecated_at": datetime.now(),
            "grace_period_ends": datetime.now() + timedelta(days=7)
        })
        
        await self.notify_user(old_key.user_id, new_key)
        
        return new_key
```

---

## Audit and Compliance

### Audit Logging

```python
class AuditLogger:
    async def log_request(
        self,
        request: LLMRequest,
        response: LLMResponse,
        auth: AuthResult
    ):
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request.id,
            "user_id": auth.user_id,
            "tenant_id": auth.tenant_id,
            "action": "llm_generate",
            "model": request.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            # Hash content for privacy
            "input_hash": self.hash_content(request.prompt),
            "output_hash": self.hash_content(response.content)
        }
        
        await self.audit_store.append(audit_entry)
```

### Compliance Reports

```python
class ComplianceReporter:
    async def generate_report(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> ComplianceReport:
        logs = await self.audit_store.query(
            tenant_id=tenant_id,
            start=start_date,
            end=end_date
        )
        
        return ComplianceReport(
            tenant_id=tenant_id,
            period=(start_date, end_date),
            total_requests=len(logs),
            unique_users=len(set(l["user_id"] for l in logs)),
            models_used=list(set(l["model"] for l in logs)),
            total_cost=sum(l["cost"] for l in logs),
            data_access_events=self.extract_data_access(logs),
            security_events=await self.get_security_events(tenant_id, start_date, end_date)
        )
```

---

## Interview Questions

### Q: How do you implement multi-tenant isolation in a RAG system?

**Strong answer:**

"Multi-tenant isolation requires defense in depth:

**Vector database level:**
- Every vector includes tenant_id in metadata
- All queries filter by tenant_id at the database level
- Never filter after retrieval (data already leaked to memory)

**Cache level:**
- All cache keys prefixed with tenant_id
- Semantic cache scoped to tenant
- No cross-tenant cache hits even for identical queries

**Prompt level:**
- Validate context documents belong to requesting tenant before including
- Never mix context from multiple tenants

**Output level:**
- Verify response does not contain cross-tenant information
- Output filtering as additional safeguard

**Audit:**
- Log all access with tenant context
- Monitor for cross-tenant access attempts

The key principle: tenant_id is a mandatory filter at every data access point, not an optional parameter."

### Q: How do you manage API keys for an LLM service?

**Strong answer:**

"Secure API key management:

**Creation:**
- Generate cryptographically random keys
- Store only the hash, return raw key once
- Associate with user, tenant, scopes, expiration

**Validation:**
- Hash incoming key, compare to stored hash
- Check expiration and revocation status
- Verify scopes match requested action

**Rotation:**
- Support key rotation with grace period
- Old key works during transition (7 days)
- Notify users of impending expiration

**Security:**
- Rate limit failed authentication attempts
- Revoke immediately on suspected compromise
- Audit all key operations

**Scopes:**
- Fine-grained: model access, operation type, daily limits
- Least privilege by default

The key principle: never store raw keys, support rotation, implement least privilege."

---

## References

- OAuth 2.0: https://oauth.net/2/
- OWASP API Security: https://owasp.org/API-Security/

---

*Previous: [Security Fundamentals](01-security-fundamentals.md)*
