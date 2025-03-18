import os
from typing import Optional, Dict, List
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.models import SecuritySchemeType
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

# Define Bearer token security scheme with better documentation
class BearerTokenAuth(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(
            scheme_name="Bearer Authentication",
            description="""
            API key authentication using Bearer token.
            
            1. In the 'Authorization' header, include: 'Bearer your-api-key'
            2. Replace 'your-api-key' with your actual API key
            
            Example: Authorization: Bearer 978e4b3fa2ab390486a0881346475e397d7a1c50a9b9f873646e6ebb7b5de33f
            """,
            auto_error=auto_error
        )

# Instantiate the security scheme
security_scheme = BearerTokenAuth(auto_error=True)

class User(BaseModel):
    """User model for authentication"""
    username: str
    is_admin: bool = False

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security_scheme)) -> User:
    """
    Validate the API key and return user information.
    
    Args:
        credentials: The bearer token credentials
        
    Returns:
        User: A User object with username and admin status
        
    Raises:
        HTTPException: If the API key is invalid
    """
    token = credentials.credentials
    
    # Validate API key
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In a real application, you might look up the user in a database
    return User(username="api_user", is_admin=True)

def require_admin(user: User = Depends(get_current_user)) -> User:
    """
    Check if the user is an admin.
    
    Args:
        user: The authenticated user
        
    Returns:
        User: The user if they are an admin
        
    Raises:
        HTTPException: If the user is not an admin
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return user 