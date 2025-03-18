from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from .api_key import API_KEY

logger = logging.getLogger("auth.middleware")

class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to handle API key authentication for all protected routes"""
    
    def __init__(self, app, public_paths=None):
        """
        Initialize the middleware with a list of public paths that don't require authentication
        
        Args:
            app: The ASGI application
            public_paths: List of paths that don't require authentication
        """
        super().__init__(app)
        self.public_paths = public_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/static"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and check for authentication
        
        Args:
            request: The incoming request
            call_next: The next middleware or endpoint handler
        
        Returns:
            The response from the next middleware or endpoint handler
        """
        # Get the path
        path = request.url.path
        
        # Skip authentication for public paths or paths starting with public paths
        if self._is_public_path(path):
            return await call_next(request)
        
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Not authenticated"},
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Extract token
        token = auth_header.replace("Bearer ", "")
        
        # Validate token
        if token != API_KEY:
            logger.warning("Invalid API key used for authentication")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid authentication credentials"},
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        try:
            # If we get here, the API key is valid
            # Add user to request state for easier access
            request.state.user = {"username": "api_user", "is_admin": True}
            return await call_next(request)
        
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Authentication error"}
            )
    
    def _is_public_path(self, path: str) -> bool:
        """
        Check if a path is public (doesn't require authentication)
        
        Args:
            path: The path to check
            
        Returns:
            bool: True if the path is public, False otherwise
        """
        # Check for exact matches first
        if path in self.public_paths:
            return True
        
        # Check for static assets
        if path.startswith("/static/"):
            return True
            
        # Check for Swagger UI paths
        if path == "/docs" or path == "/redoc" or path == "/openapi.json":
            return True
            
        # Check for Swagger UI related paths
        if path.startswith("/docs/") or path.startswith("/redoc/") or path.startswith("/openapi/"):
            return True
            
        # Check if path starts with any public path
        for public_path in self.public_paths:
            if path.startswith(public_path + "/"):
                return True
                
        return False 