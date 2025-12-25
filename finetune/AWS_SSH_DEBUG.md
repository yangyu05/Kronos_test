# Debug: AWS EC2 SSH "Permission denied (publickey)" Error

## Common Causes and Solutions

### 1. Check Key File Permissions (Most Common Issue)

On your **local machine** (Mac/Linux), the key file must have correct permissions:

```bash
# Check current permissions
ls -l your-key.pem

# Fix permissions (should be 400 or 600)
chmod 400 your-key.pem

# Or
chmod 600 your-key.pem

# Verify
ls -l your-key.pem
# Should show: -r-------- or -rw-------
```

**Windows users**: If using WSL or Git Bash, use the same commands.

### 2. Verify Correct Username

Different AWS AMIs use different default usernames:

```bash
# Ubuntu AMI
ssh -i your-key.pem ubuntu@your-instance-ip

# Amazon Linux 2
ssh -i your-key.pem ec2-user@your-instance-ip

# Amazon Linux 2023
ssh -i your-key.pem ec2-user@your-instance-ip

# Debian
ssh -i your-key.pem admin@your-instance-ip

# RHEL/CentOS
ssh -i your-key.pem ec2-user@your-instance-ip

# SUSE
ssh -i your-key.pem ec2-user@your-instance-ip
```

**To find your AMI username:**
1. Go to EC2 Console → Instances → Select your instance
2. Check the "Details" tab → "AMI" → Click on the AMI link
3. Check the AMI description for the default username

### 3. Verify Key Pair is Correct

Make sure you're using the key pair that was selected when launching the instance:

```bash
# Check which key pair is associated with your instance
# In AWS Console: EC2 → Instances → Select instance → Details tab → "Key pair name"
```

### 4. Verify Instance IP Address

Make sure you're using the correct IP:

```bash
# Public IPv4 address (for instances with public IP)
# Found in: EC2 Console → Instances → Select instance → Details tab

# Or use Public DNS name
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute-1.amazonaws.com
```

### 5. Check Security Group

Ensure your security group allows SSH (port 22):

1. Go to EC2 Console → Instances → Select your instance
2. Click "Security" tab → Click on the security group
3. Check "Inbound rules" → Should have:
   - Type: SSH
   - Protocol: TCP
   - Port: 22
   - Source: Your IP or 0.0.0.0/0 (less secure)

### 6. Verify Instance is Running

```bash
# In AWS Console, check instance state
# Should be "running" (green circle)
```

### 7. Try Verbose SSH for More Details

```bash
# Use verbose mode to see what's happening
ssh -v -i your-key.pem ubuntu@your-instance-ip

# Even more verbose
ssh -vv -i your-key.pem ubuntu@your-instance-ip

# Maximum verbosity
ssh -vvv -i your-key.pem ubuntu@your-instance-ip
```

This will show you exactly where the connection is failing.

## Step-by-Step Debugging Process

### Step 1: Verify Key File Exists and Has Correct Permissions

```bash
# On your local machine
cd ~/Downloads  # or wherever your key file is
ls -l *.pem

# Fix permissions if needed
chmod 400 your-key.pem
```

### Step 2: Get Instance Information from AWS Console

1. Go to EC2 Console → Instances
2. Select your instance
3. Note down:
   - **Public IPv4 address** (or Public DNS name)
   - **Key pair name**
   - **AMI** (to determine username)
   - **State** (should be "running")

### Step 3: Determine Correct Username

Based on your AMI:
- **Ubuntu**: `ubuntu`
- **Amazon Linux**: `ec2-user`
- **Debian**: `admin`
- **RHEL/CentOS**: `ec2-user`

Or check the AMI description in the console.

### Step 4: Test Connection

```bash
# Replace with your actual values
ssh -i /path/to/your-key.pem USERNAME@PUBLIC_IP

# Example for Ubuntu:
ssh -i ~/Downloads/my-key.pem ubuntu@54.123.45.67
```

### Step 5: If Still Failing, Check Security Group

1. EC2 Console → Instances → Select instance
2. Security tab → Click security group name
3. Inbound rules → Edit inbound rules
4. Add rule if missing:
   - Type: SSH
   - Source: My IP (or 0.0.0.0/0 for testing)
   - Save rules

## Alternative: Use AWS Systems Manager Session Manager

If SSH still doesn't work, you can use AWS Session Manager (no key needed):

### Prerequisites:
1. Install AWS CLI: `pip install awscli` or `brew install awscli`
2. Install Session Manager plugin: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html
3. Instance must have SSM agent (most AMIs have it)

### Connect:
```bash
aws ssm start-session --target i-1234567890abcdef0
```

Replace `i-1234567890abcdef0` with your instance ID.

## Quick Checklist

- [ ] Key file permissions are 400 or 600
- [ ] Using correct username (ubuntu/ec2-user/admin)
- [ ] Using correct key pair (matches instance)
- [ ] Using correct IP address
- [ ] Security group allows SSH (port 22)
- [ ] Instance is in "running" state
- [ ] Key file path is correct (use absolute path if needed)

## Common Mistakes

1. **Wrong username**: Most common with Ubuntu vs Amazon Linux
2. **Wrong key file**: Using a different key than the one associated with instance
3. **Permissions**: Key file must be 400/600, not 644 or 777
4. **Wrong IP**: Using private IP instead of public IP
5. **Security group**: Not allowing SSH from your IP

## Still Not Working?

Try these additional steps:

### Option 1: Create New Key Pair and Add to Instance

1. Create new key pair in EC2 Console
2. Download the .pem file
3. Set permissions: `chmod 400 new-key.pem`
4. Use AWS Systems Manager to add the new key to the instance

### Option 2: Use EC2 Instance Connect (Browser-based)

1. EC2 Console → Instances → Select instance
2. Click "Connect" button
3. Choose "EC2 Instance Connect" tab
4. Click "Connect" (opens browser-based terminal)

### Option 3: Check Instance Logs

1. EC2 Console → Instances → Select instance
2. Actions → Monitor and troubleshoot → Get system log
3. Check for any errors

## Example Correct SSH Command

```bash
# Ubuntu instance
ssh -i ~/Downloads/my-aws-key.pem ubuntu@54.123.45.67

# Amazon Linux instance  
ssh -i ~/Downloads/my-aws-key.pem ec2-user@54.123.45.67

# With verbose output for debugging
ssh -vv -i ~/Downloads/my-aws-key.pem ubuntu@54.123.45.67
```

