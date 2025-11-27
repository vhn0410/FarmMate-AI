"""
Keycloak JWT Authentication for FastAPI
"""
import os
from typing import Optional
from fastapi import HTTPException, Security, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt import PyJWKClient
from functools import lru_cache

# Keycloak Configuration
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
REALM = os.getenv("KEYCLOAK_REALM", "your-realm")
CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "your-client-id")

# JWKS URL for token verification
JWKS_URL = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/certs"

# Security scheme
security = HTTPBearer()


@lru_cache()
def get_jwks_client():
    """Cache JWKS client to avoid repeated network calls"""
    return PyJWKClient(JWKS_URL)


class KeycloakUser:
    """Parsed user information from JWT token"""
    def __init__(self, token_data: dict):
        self.sub = token_data.get("sub")  # User ID
        self.email = token_data.get("email")
        self.preferred_username = token_data.get("preferred_username")
        self.name = token_data.get("name")
        self.realm_roles = token_data.get("realm_access", {}).get("roles", [])
        self.client_roles = token_data.get("resource_access", {}).get(CLIENT_ID, {}).get("roles", [])
        self.raw_token = token_data

    def has_role(self, role: str) -> bool:
        """Check if user has a specific realm role"""
        return role in self.realm_roles

    def has_client_role(self, role: str) -> bool:
        """Check if user has a specific client role"""
        return role in self.client_roles


def verify_token(token: str) -> dict:
    """
    Verify JWT token from Keycloak
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token data
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        jwks_client = get_jwks_client()
        
        # Get signing key from JWKS
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        # Verify and decode token
        decoded_token = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=CLIENT_ID,  # Validate audience
            options={
                "verify_signature": True,
                "verify_aud": True,
                "verify_exp": True
            }
        )
        
        return decoded_token
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Token verification failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> KeycloakUser:
    """
    FastAPI dependency to get current authenticated user
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: KeycloakUser = Depends(get_current_user)):
            return {"user_id": user.sub, "username": user.preferred_username}
    """
    token = credentials.credentials
    token_data = verify_token(token)
    return KeycloakUser(token_data)


async def get_optional_user(
    authorization: Optional[str] = Header(None)
) -> Optional[KeycloakUser]:
    """
    Optional authentication - returns None if no token provided
    
    Usage:
        @app.get("/optional")
        async def optional_route(user: Optional[KeycloakUser] = Depends(get_optional_user)):
            if user:
                return {"authenticated": True, "user_id": user.sub}
            return {"authenticated": False}
    """
    if authorization is None:
        return None
    
    # Extract token from "Bearer <token>"
    if not authorization.startswith("Bearer "):
        return None
    
    token = authorization.replace("Bearer ", "")
    
    try:
        token_data = verify_token(token)
        return KeycloakUser(token_data)
    except HTTPException:
        return None


def require_role(required_role: str):
    """
    Dependency factory for role-based access control
    
    Usage:
        @app.get("/admin")
        async def admin_route(user: KeycloakUser = Depends(require_role("admin"))):
            return {"message": "Admin access granted"}
    """
    def role_checker(user: KeycloakUser = Depends(get_current_user)) -> KeycloakUser:
        if not user.has_role(required_role):
            raise HTTPException(
                status_code=403,
                detail=f"User does not have required role: {required_role}"
            )
        return user
    
    return role_checker


def require_client_role(required_role: str):
    """
    Dependency factory for client role-based access control
    
    Usage:
        @app.get("/client-protected")
        async def client_protected(user: KeycloakUser = Depends(require_client_role("app-user"))):
            return {"message": "Client role verified"}
    """
    def client_role_checker(user: KeycloakUser = Depends(get_current_user)) -> KeycloakUser:
        if not user.has_client_role(required_role):
            raise HTTPException(
                status_code=403,
                detail=f"User does not have required client role: {required_role}"
            )
        return user
    
    return client_role_checker